from __future__ import annotations

from sklearn.linear_model import Lasso

from euclid.search.engine_contracts import EngineInputContext, EngineRunResult
from euclid.search.engines._linear_candidate import (
    engine_result,
    linear_candidate_record,
    linear_expression,
    numeric_training_data,
    result_for_insufficient_data,
    rounded_mapping,
    stable_int_seed,
)


class SparseRegressionEngine:
    engine_id = "sparse-regression-engine-v1"
    engine_version = "1.0"

    def __init__(
        self,
        *,
        alpha: float = 1.0e-6,
        coefficient_threshold: float = 1.0e-8,
        max_iter: int = 10000,
    ) -> None:
        self.alpha = float(alpha)
        self.coefficient_threshold = float(coefficient_threshold)
        self.max_iter = int(max_iter)

    def run(self, context: EngineInputContext) -> EngineRunResult:
        data = numeric_training_data(context)
        if data.y.size < 2 or data.x.shape[1] == 0:
            return result_for_insufficient_data(
                context=context,
                engine_id=self.engine_id,
                engine_version=self.engine_version,
                reason_code="insufficient_sparse_regression_inputs",
                message=(
                    "sparse regression engine requires at least two numeric rows "
                    "and one numeric feature"
                ),
                details={
                    "numeric_row_count": int(data.y.size),
                    "feature_count": int(data.x.shape[1]),
                    "omitted_row_count": data.omitted_row_count,
                },
            )

        model = Lasso(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=self.max_iter,
            selection="cyclic",
            random_state=stable_int_seed(context.random_seed),
        )
        model.fit(data.x, data.y)
        coefficients = {
            name: float(value)
            for name, value in zip(data.feature_names, model.coef_)
        }
        active_coefficients = {
            name: value
            for name, value in coefficients.items()
            if abs(value) > self.coefficient_threshold
        }
        expression = linear_expression(
            intercept=float(model.intercept_),
            coefficients=active_coefficients,
            coefficient_threshold=self.coefficient_threshold,
        )
        predictions = model.predict(data.x)
        residual_sum_squares = float(((data.y - predictions) ** 2).sum())
        trace = {
            "sparse_regression_backend": "sklearn.linear_model.Lasso",
            "regularization_alpha": self.alpha,
            "coefficient_threshold": self.coefficient_threshold,
            "max_iter": self.max_iter,
            "n_iter": int(model.n_iter_),
            "intercept": float(round(float(model.intercept_), 12)),
            "coefficients": rounded_mapping(coefficients),
            "active_support": tuple(sorted(active_coefficients)),
            "active_feature_count": len(active_coefficients),
            "residual_sum_squares": float(round(residual_sum_squares, 12)),
            "numeric_row_count": int(data.y.size),
        }
        omission = {
            "omitted_non_numeric_rows": data.omitted_row_count,
            "inactive_features": sorted(
                set(data.feature_names).difference(active_coefficients)
            ),
        }
        record = linear_candidate_record(
            context=context,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            candidate_id=f"sparse-regression-{context.search_plan_id}-00",
            proposal_rank=0,
            expression=expression,
            active_feature_names=active_coefficients.keys(),
            rows_used=data.rows_used,
            search_space_declaration="sklearn_lasso_sparse_linear_support_v1",
            cir_family_id="analytic",
            cir_form_class="sparse_linear_expression_ir",
            backend_family="sklearn_lasso_sparse_regression",
            candidate_trace=trace,
            omission_disclosure=omission,
        )
        return engine_result(
            context=context,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            records=(record,),
            trace=trace,
            omission_disclosure=omission,
        )


__all__ = ["SparseRegressionEngine"]
