from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

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


class LatentStateEngine:
    engine_id = "latent-state-engine-v1"
    engine_version = "1.0"

    def run(self, context: EngineInputContext) -> EngineRunResult:
        data = numeric_training_data(context)
        if data.y.size < 2 or data.x.shape[1] == 0:
            return result_for_insufficient_data(
                context=context,
                engine_id=self.engine_id,
                engine_version=self.engine_version,
                reason_code="insufficient_latent_state_inputs",
                message=(
                    "latent-state engine requires at least two numeric rows and "
                    "one numeric feature"
                ),
                details={
                    "numeric_row_count": int(data.y.size),
                    "feature_count": int(data.x.shape[1]),
                    "omitted_row_count": data.omitted_row_count,
                },
            )

        means = data.x.mean(axis=0)
        scales = data.x.std(axis=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        standardized = (data.x - means) / scales
        pca = PCA(
            n_components=1,
            svd_solver="full",
            random_state=stable_int_seed(context.random_seed),
        )
        latent_scores = pca.fit_transform(standardized)[:, 0]
        design = np.column_stack((np.ones_like(latent_scores), latent_scores))
        solution, residuals, rank, singular_values = linalg.lstsq(design, data.y)
        latent_intercept = float(solution[0])
        latent_slope = float(solution[1])
        component = pca.components_[0]
        feature_coefficients = {
            name: float(latent_slope * loading / scale)
            for name, loading, scale in zip(data.feature_names, component, scales)
        }
        intercept = float(
            latent_intercept
            - sum(
                latent_slope * loading * mean / scale
                for loading, mean, scale in zip(component, means, scales)
            )
        )
        active_coefficients = {
            name: value
            for name, value in feature_coefficients.items()
            if abs(value) > 1e-10
        }
        expression = linear_expression(
            intercept=intercept,
            coefficients=active_coefficients,
            coefficient_threshold=1e-10,
        )
        trace = {
            "latent_state_backend": "sklearn.decomposition.PCA",
            "projection_regression_backend": "scipy.linalg.lstsq",
            "latent_component_count": 1,
            "explained_variance_ratio": [
                float(round(value, 12)) for value in pca.explained_variance_ratio_
            ],
            "component_loadings": rounded_mapping(
                {
                    name: float(loading)
                    for name, loading in zip(data.feature_names, component)
                }
            ),
            "latent_intercept": float(round(latent_intercept, 12)),
            "latent_slope": float(round(latent_slope, 12)),
            "proxy_intercept": float(round(intercept, 12)),
            "proxy_feature_coefficients": rounded_mapping(active_coefficients),
            "matrix_rank": int(rank),
            "singular_values": [float(round(value, 12)) for value in singular_values],
            "residual_sum_squares": float(round(float(residuals.sum()), 12))
            if residuals.size
            else 0.0,
            "numeric_row_count": int(data.y.size),
        }
        omission = {
            "omitted_non_numeric_rows": data.omitted_row_count,
            "latent_state_observability": "one_factor_linear_proxy",
        }
        record = linear_candidate_record(
            context=context,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            candidate_id=f"latent-state-{context.search_plan_id}-00",
            proposal_rank=0,
            expression=expression,
            active_feature_names=active_coefficients.keys(),
            rows_used=data.rows_used,
            search_space_declaration="sklearn_pca_one_factor_proxy_v1",
            cir_family_id="analytic",
            cir_form_class="latent_state_proxy_expression_ir",
            backend_family="sklearn_pca_latent_state_proxy",
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


__all__ = ["LatentStateEngine"]
