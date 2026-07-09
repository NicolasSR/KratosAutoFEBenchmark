from dataclasses import dataclass

import numpy as np
import sympy as sp

@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

class GroundTruthSolver():

    # Ground truth solution parameters
    kappa = 1
    a = 1
    b = 1
    L = 5
    H = 1
    sigma = 0

    def __init__(self, mdpa_filename):
        self.nodes, self.max_element_size, self.bounds = self.extract_nodes_element_size_and_bounds(mdpa_filename)

        # Ckeck that the bounds correspond to the domain size given by the parameters L and H
        assert np.isclose(self.bounds.x_min, 0.0, rtol=1e-10), f"Expected x_min to be 0.0, but got {self.bounds.x_min}"
        assert np.isclose(self.bounds.x_max, self.L, rtol=1e-10), f"Expected x_max to be {self.L}, but got {self.bounds.x_max}"
        assert np.isclose(self.bounds.y_min, -self.H, rtol=1e-10), f"Expected y_min to be {-self.H}, but got {self.bounds.y_min}"
        assert np.isclose(self.bounds.y_max, self.H, rtol=1e-10), f"Expected y_max to be {self.H}, but got {self.bounds.y_max}"

        self.grad_u_expr = self.precompute_grad_u_expr()

    def get_sigma_parameter(self):
        return self.sigma

    def extract_nodes_element_size_and_bounds(self, file_path):
        nodes = {}
        max_element_size = 0.0

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        in_nodes = False
        
        with open(file_path, 'r') as f:
            for line in f:
                # Detect the Node block
                if "Begin Nodes" in line:
                    in_nodes = True
                    continue
                if "End Nodes" in line:
                    break
                
                if in_nodes:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_id = int(parts[0])
                            x, y = float(parts[1]), float(parts[2])
                            
                            # Store in dictionary
                            nodes[node_id] = np.array([x, y])
                            
                            # Update bounds
                            min_x, max_x = min(min_x, x), max(max_x, x)
                            min_y, max_y = min(min_y, y), max(max_y, y)
                        except ValueError:
                            continue

        with open(file_path, 'r') as f:
            for line in f:
                # Detect the Elements block
                if "Begin Elements" in line:
                    in_nodes = True
                    continue
                if "End Elements" in line:
                    break
                
                if in_nodes:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            # Store in dictionary

                            max_element_size = max(max_element_size, np.linalg.norm(nodes[int(parts[2])]-nodes[int(parts[3])]))
                            max_element_size = max(max_element_size, np.linalg.norm(nodes[int(parts[3])]-nodes[int(parts[4])]))
                            max_element_size = max(max_element_size, np.linalg.norm(nodes[int(parts[4])]-nodes[int(parts[2])]))

                        except ValueError:
                            continue

        bounds = Bounds(min_x, max_x, min_y, max_y)

        return nodes, max_element_size, bounds
    
    def compute_delta_parameter(self):
        gamma = 1
        a = self.a
        b = self.b
        y = sp.Symbol('y', real=True)
        nu_expr = a*abs(y)+b
        y_interval = sp.Interval(self.bounds.y_min,self.bounds.y_max)
        nu_min = float(sp.minimum(nu_expr,y,y_interval))
        nu_max = float(sp.maximum(nu_expr,y,y_interval))
        deriv_nu_y_expr = sp.diff(nu_expr).doit()
        if deriv_nu_y_expr.has(sp.sign(y)):
            nu_grad_max_norm = sp.maximum(deriv_nu_y_expr.subs(sp.sign(y),-1),y,y_interval)
            nu_grad_max_norm = max(nu_grad_max_norm, sp.maximum(deriv_nu_y_expr.subs(sp.sign(y),1),y,y_interval))
        else:
            nu_grad_max_norm = sp.maximum(deriv_nu_y_expr,y,y_interval)
        nu_grad_max_norm=float(nu_grad_max_norm)
        h = self.max_element_size

        delta = gamma*nu_min*h**2/12*1/(h**2*nu_grad_max_norm**2+nu_max**2)

        return delta/40
        # return delta

    
    def compute_ground_truth_at_points(self, x_vec, y_vec):

        kappa = self.kappa
        a = self.a
        b = self.b
        L = self.L
        H = self.H

        u_x = kappa/a*(H-abs(y_vec)+b/a*np.log((a*abs(y_vec)+b)/(a*H+b)))
        u_y = 0

        p = kappa*(L-x_vec)

        return u_x, u_y, p
    
    def compute_ground_truth_at_nodes(self):
        node_coords = list(self.nodes.values())
        x_vec = np.array([coord[0] for coord in node_coords])
        y_vec = np.array([coord[1] for coord in node_coords])

        print(self.compute_ground_truth_at_points(x_vec, y_vec))

    def get_dirichlet_boundary_analytical_functions(self):
        # In the inlet condition within Kratos, we may input the dirichlet boundary condition values
        # as a function of x and y.

        kappa = self.kappa
        a = self.a
        b = self.b
        L = self.L
        H = self.H

        u_x = f"{kappa}/{a}*({H}-abs(y)+{b}/{a}*ln(({a}*abs(y)+{b})/({a}*{H}+{b})))"
        u_y = 0
        p = f"{kappa}*({L}-x)"

        return u_x, u_y, p
    
    def compute_viscosity_at_points(self, x, y):
        return self.a*abs(y)+self.b
    
    def precompute_grad_u_expr(self):
        x, y = sp.symbols("x y", real=True)

        kappa = self.kappa
        a = self.a
        b = self.b
        L = self.L
        H = self.H

        u_x = kappa/a*(H-abs(y)+b/a*sp.log((a*abs(y)+b)/(a*H+b)))
        u_y = 0
        u = sp.Array([u_x,u_y])

        grad_u_expr = sp.derive_by_array(u, sp.Array([x,y]))
        print("Analytical expression of the gradient of u:", grad_u_expr)
        return grad_u_expr
    
    def compute_p_and_grad_u_at_point(self, x_val,y_val):

        _, _, p = self.compute_ground_truth_at_points(x_val, y_val)

        x, y = sp.symbols("x y", real=True)
        local_grad_u_expr = self.grad_u_expr
        local_grad_u_expr = local_grad_u_expr.subs(x, x_val).doit()
        local_grad_u_expr = local_grad_u_expr.subs(y, y_val).doit()
        grad_u = np.array(local_grad_u_expr, dtype=float)

        return p, grad_u
    

    
