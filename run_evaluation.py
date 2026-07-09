import importlib

case_name = "first_order_stokes_variable_viscosity_bvs_gl"


evaluation_module = importlib.import_module(f"kratos_simulation_tests.{case_name}.evaluation_script")

evaluation_module.run_evaluation()