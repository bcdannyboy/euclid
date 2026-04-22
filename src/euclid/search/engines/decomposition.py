from __future__ import annotations

import numpy as np
from scipy import linalg

from euclid.search.engine_contracts import EngineInputContext, EngineRunResult
from euclid.search.engines._linear_candidate import (
    engine_result,
    linear_candidate_record,
    linear_expression,
    numeric_training_data,
    result_for_insufficient_data,
    rounded_mapping,
)


class DecompositionEngine:
    engine_id = "decomposition-engine-v1"
    engine_version = "1.0"

    def run(self, context: EngineInputContext) -> EngineRunResult:
        data = numeric_training_data(context)
        if data.y.size < 2:
            return result_for_insufficient_data(
                context=context,
                engine_id=self.engine_id,
                engine_version=self.engine_version,
                reason_code="insufficient_numeric_rows",
                message=(
                    "decomposition engine requires at least two numeric target rows"
                ),
                details={
                    "numeric_row_count": int(data.y.size),
                    "omitted_row_count": data.omitted_row_count,
                },
            )

        intercept_column = data.x[:, :0] + 1.0
        design = (
            intercept_column
            if data.x.shape[1] == 0
            else np.column_stack((intercept_column, data.x))
        )
        solution, residuals, rank, singular_values = linalg.lstsq(design, data.y)
        intercept = float(solution[0])
        coefficients = {
            feature_name: float(value)
            for feature_name, value in zip(data.feature_names, solution[1:])
        }
        active_coefficients = {
            name: value for name, value in coefficients.items() if abs(value) > 1e-10
        }
        expression = linear_expression(
            intercept=intercept,
            coefficients=active_coefficients,
            coefficient_threshold=1e-10,
        )
        candidate_id = f"decomposition-{context.search_plan_id}-00"
        trace = {
            "decomposition_backend": "scipy.linalg.lstsq",
            "matrix_rank": int(rank),
            "singular_values": [float(round(value, 12)) for value in singular_values],
            "residual_sum_squares": float(round(float(residuals.sum()), 12))
            if residuals.size
            else 0.0,
            "intercept": float(round(intercept, 12)),
            "component_coefficients": rounded_mapping(active_coefficients),
            "numeric_row_count": int(data.y.size),
        }
        omission = {
            "omitted_non_numeric_rows": data.omitted_row_count,
            "candidate_family": "additive_linear_decomposition",
        }
        record = linear_candidate_record(
            context=context,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            candidate_id=candidate_id,
            proposal_rank=0,
            expression=expression,
            active_feature_names=active_coefficients.keys(),
            rows_used=data.rows_used,
            search_space_declaration="scipy_lstsq_additive_decomposition_v1",
            cir_family_id="analytic",
            cir_form_class="additive_decomposition_expression_ir",
            backend_family="scipy_lstsq_additive_decomposition",
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


__all__ = ["DecompositionEngine"]
