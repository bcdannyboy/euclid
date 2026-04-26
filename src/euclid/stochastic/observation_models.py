from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from scipy import special, stats

from euclid.contracts.errors import ContractValidationError


@dataclass(frozen=True)
class ObservationModelSpec:
    family_id: str

    def bind(self, parameters: Mapping[str, float]) -> "BoundObservationModel":
        return BoundObservationModel(
            family_id=str(self.family_id),
            parameters={key: float(value) for key, value in parameters.items()},
        )


@dataclass(frozen=True)
class BoundObservationModel:
    family_id: str
    parameters: Mapping[str, float]

    @property
    def distribution_backend(self) -> str:
        return "scipy.stats"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return _REQUIRED_PARAMETERS[self.family_id]

    @property
    def distribution_family_id(self) -> str:
        return _DISTRIBUTION_FAMILY_IDS[self.family_id]

    def support_contains(self, value: float) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
        family = self.family_id
        if family in {"gaussian", "student_t", "laplace"}:
            return True
        if family in {"poisson", "negative_binomial"}:
            return numeric >= 0 and numeric.is_integer()
        if family == "bernoulli":
            return numeric in {0.0, 1.0}
        if family == "beta":
            return 0.0 <= numeric <= 1.0
        if family == "lognormal":
            return numeric > 0.0
        return False

    def log_likelihood(self, value: float) -> float:
        self._validate_parameters()
        observed = float(value)
        if not self.support_contains(observed):
            raise ContractValidationError(
                code="observation_outside_model_support",
                message="observation is outside the declared stochastic support",
                field_path="observation",
                details={"family_id": self.family_id, "value": observed},
            )
        family = self.family_id
        if family == "beta":
            if observed in {0.0, 1.0}:
                raise ContractValidationError(
                    code="observation_outside_model_support",
                    message="beta likelihood requires an open-unit observation",
                    field_path="observation",
                    details={"value": observed},
                )
        distribution = self._scipy_distribution()
        scorer = distribution.logpmf if family in _DISCRETE_FAMILIES else distribution.logpdf
        likelihood = float(scorer(observed))
        if not math.isfinite(likelihood):
            raise ContractValidationError(
                code="nonfinite_likelihood",
                message="stochastic model likelihood must be finite",
                field_path="observation",
                details={"family_id": family, "value": observed},
            )
        return likelihood

    def cdf(self, value: float) -> float:
        self._validate_parameters()
        observed = float(value)
        probability = float(self._scipy_distribution().cdf(observed))
        return max(0.0, min(1.0, probability))

    def ppf(self, probability: float) -> float:
        self._validate_parameters()
        quantile_probability = float(probability)
        if not 0.0 < quantile_probability < 1.0:
            raise ContractValidationError(
                code="invalid_probability_level",
                message="probability levels must be strictly between 0 and 1",
                field_path="probability",
                details={"probability": probability},
            )
        value = float(self._scipy_distribution().ppf(quantile_probability))
        if not math.isfinite(value):
            raise ContractValidationError(
                code="nonfinite_stochastic_quantile",
                message="stochastic quantile helper produced a nonfinite value",
                field_path="probability",
                details={"probability": probability},
            )
        return value

    def interval(self, nominal_coverage: float) -> tuple[float, float]:
        coverage = float(nominal_coverage)
        if not 0.0 < coverage < 1.0:
            raise ContractValidationError(
                code="invalid_interval_coverage",
                message="interval coverage must be strictly between 0 and 1",
                field_path="nominal_coverage",
                details={"nominal_coverage": nominal_coverage},
            )
        tail = (1.0 - coverage) / 2.0
        return self.ppf(tail), self.ppf(1.0 - tail)

    def pit(
        self,
        value: float,
        *,
        randomized: bool = False,
        row_key: str | None = None,
        randomization_seed: str = "0",
    ) -> float:
        self._validate_parameters()
        observed = float(value)
        if self.family_id not in _DISCRETE_FAMILIES:
            return self.cdf(observed)
        if not randomized:
            raise ContractValidationError(
                code="unsupported_pit_family",
                message=(
                    "discrete observation families require deterministic "
                    "randomized PIT"
                ),
                field_path="pit.randomized",
                details={"family_id": self.family_id},
            )
        if row_key is None or not str(row_key):
            raise ContractValidationError(
                code="missing_pit_randomization_key",
                message="randomized PIT requires a stable row key",
                field_path="pit.row_key",
            )
        distribution = self._scipy_distribution()
        lower = float(distribution.cdf(observed - 1.0))
        upper = float(distribution.cdf(observed))
        draw = _deterministic_unit_interval(
            f"{randomization_seed}:{row_key}:{self.family_id}:{observed}"
        )
        probability = lower + (draw * max(upper - lower, 0.0))
        return max(0.0, min(1.0, probability))

    def survival(self, value: float) -> float:
        self._validate_parameters()
        probability = float(self._scipy_distribution().sf(float(value)))
        return max(0.0, min(1.0, probability))

    def _scipy_distribution(self):
        parameters = self.parameters
        family = self.family_id
        if family == "gaussian":
            return stats.norm(
                loc=parameters["location"],
                scale=parameters["scale"],
            )
        if family == "student_t":
            return stats.t(
                df=parameters["df"],
                loc=parameters["location"],
                scale=parameters["scale"],
            )
        if family == "laplace":
            return stats.laplace(
                loc=parameters["location"],
                scale=parameters["scale"],
            )
        if family == "poisson":
            return stats.poisson(mu=parameters["rate"])
        if family == "negative_binomial":
            mean = parameters["mean"]
            dispersion = parameters["dispersion"]
            probability = dispersion / (dispersion + mean)
            return stats.nbinom(n=dispersion, p=probability)
        if family == "bernoulli":
            return stats.bernoulli(p=parameters["probability"])
        if family == "beta":
            return stats.beta(a=parameters["alpha"], b=parameters["beta"])
        if family == "lognormal":
            return stats.lognorm(
                s=parameters["log_scale"],
                scale=math.exp(parameters["log_location"]),
            )
        raise ContractValidationError(
            code="unsupported_observation_family",
            message="unsupported stochastic observation family",
            field_path="family_id",
            details={"family_id": family},
        )

    def _validate_parameters(self) -> None:
        required = _REQUIRED_PARAMETERS.get(self.family_id)
        if required is None:
            raise ContractValidationError(
                code="unsupported_observation_family",
                message="unsupported stochastic observation family",
                field_path="family_id",
                details={"family_id": self.family_id},
            )
        missing = [name for name in required if name not in self.parameters]
        if missing:
            raise ContractValidationError(
                code="missing_stochastic_parameter",
                message="stochastic model parameters are incomplete",
                field_path="parameters",
                details={"missing": missing},
            )
        nonfinite = [
            name
            for name in required
            if not math.isfinite(float(self.parameters[name]))
        ]
        if nonfinite:
            raise ContractValidationError(
                code="nonfinite_likelihood",
                message="stochastic model parameters must be finite",
                field_path="parameters",
                details={"nonfinite_parameters": nonfinite},
            )
        parameters = self.parameters
        if self.family_id in {"gaussian", "student_t", "laplace"} and parameters[
            "scale"
        ] <= 0:
            _raise_invalid("scale", parameters["scale"])
        if self.family_id == "student_t" and parameters["df"] <= 2:
            _raise_invalid("df", parameters["df"])
        if self.family_id == "poisson" and parameters["rate"] <= 0:
            _raise_invalid("rate", parameters["rate"])
        if self.family_id == "negative_binomial" and (
            parameters["mean"] <= 0 or parameters["dispersion"] <= 0
        ):
            _raise_invalid("negative_binomial", 0.0)
        if self.family_id == "bernoulli" and not (0 <= parameters["probability"] <= 1):
            _raise_invalid("probability", parameters["probability"])
        if self.family_id == "beta" and (
            parameters["alpha"] <= 0 or parameters["beta"] <= 0
        ):
            _raise_invalid("beta", 0.0)
        if self.family_id == "lognormal" and parameters["log_scale"] <= 0:
            _raise_invalid("log_scale", parameters["log_scale"])


@dataclass(frozen=True)
class MixtureObservationModel:
    components: Sequence[BoundObservationModel]
    weights: Sequence[float]
    family_id: str = "mixture"

    def log_likelihood(self, value: float) -> float:
        self._validate()
        terms = [
            math.log(float(weight)) + component.log_likelihood(value)
            for component, weight in zip(self.components, self.weights, strict=True)
        ]
        total = float(special.logsumexp(terms))
        if not math.isfinite(total):
            raise ContractValidationError(
                code="nonfinite_likelihood",
                message="mixture likelihood must be finite",
                field_path="mixture",
            )
        return total

    def pit(self, value: float) -> float:
        del value
        raise ContractValidationError(
            code="unsupported_pit_family",
            message="mixture PIT is not admitted without an explicit tested policy",
            field_path="pit.family_id",
            details={"family_id": self.family_id},
        )

    def _validate(self) -> None:
        if not self.components or len(self.components) != len(self.weights):
            raise ContractValidationError(
                code="invalid_mixture_model",
                message="mixture model requires matching components and weights",
                field_path="mixture",
            )
        total = sum(float(weight) for weight in self.weights)
        if any(float(weight) <= 0 for weight in self.weights) or not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractValidationError(
                code="invalid_mixture_weight_simplex",
                message="mixture weights must be a positive simplex",
                field_path="weights",
                details={"weight_sum": total},
            )


_DISCRETE_FAMILIES = frozenset({"poisson", "negative_binomial", "bernoulli"})

_REQUIRED_PARAMETERS = {
    "gaussian": ("location", "scale"),
    "student_t": ("location", "scale", "df"),
    "laplace": ("location", "scale"),
    "poisson": ("rate",),
    "negative_binomial": ("mean", "dispersion"),
    "bernoulli": ("probability",),
    "beta": ("alpha", "beta"),
    "lognormal": ("log_location", "log_scale"),
}

_DISTRIBUTION_FAMILY_IDS = {
    "gaussian": "gaussian_location_scale",
    "student_t": "student_t_location_scale",
    "laplace": "laplace_location_scale",
    "poisson": "poisson_rate",
    "negative_binomial": "negative_binomial_mean_dispersion",
    "bernoulli": "bernoulli_probability",
    "beta": "beta_alpha_beta",
    "lognormal": "lognormal_location_scale",
}


def get_observation_model(family_id: str) -> ObservationModelSpec:
    family = str(family_id)
    if family not in {
        "gaussian",
        "student_t",
        "laplace",
        "poisson",
        "negative_binomial",
        "bernoulli",
        "beta",
        "lognormal",
    }:
        raise ContractValidationError(
            code="unsupported_observation_family",
            message="unsupported stochastic observation family",
            field_path="family_id",
            details={"family_id": family},
        )
    return ObservationModelSpec(family)


def _raise_invalid(name: str, value: float) -> None:
    raise ContractValidationError(
        code="invalid_stochastic_parameter",
        message="stochastic model parameter is outside its admissible domain",
        field_path=f"parameters.{name}",
        details={"value": value},
    )


def _deterministic_unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


__all__ = [
    "BoundObservationModel",
    "MixtureObservationModel",
    "ObservationModelSpec",
    "get_observation_model",
]
