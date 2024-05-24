function [x, tab_r] = bi_conjugate_gradient(A,b,x0,tol,itermax)
  %Bi-conjugate gradient algorithm
  %- x0: starting vector
  %- tol: accuracy
  %- itermax: maximum iterations number before stopping algorithm
  %- x: solution founded
  %- tab_r: residuals vector
  n = length(A);
  x_new = zeros(n,1);
  x_old = x0;
  r_old = b - A * x0;
  r_tilde_old = r_old;
  tab_r(1) = norm(r_old);
  p_old = r_old;
  p_tilde_old = r_tilde_old;
  for k=0:itermax
    Ap = A * p_old;
    r_old_r_tilde_old = r_old' * r_tilde_old;
    alpha = r_old_r_tilde_old/(Ap' * p_tilde_old);
    x_new = x_old + (alpha * p_old);
    r_new = r_old - alpha * Ap;
    r_tilde_new = r_tilde_old - alpha * A' * p_tilde_old;
    beta = (r_new' * r_tilde_new)/r_old_r_tilde_old;
    p_new = r_new + beta * p_old;
    p_tilde_new = r_tilde_new + beta * p_tilde_old;
    tab_r(k+2) = norm(r_new);
    if norm(r_new) < tol
      x = x_new;
      iter = k;
      return;
    else
      r_old = r_new;
      r_tilde_old = r_tilde_new;
      p_old = p_new;
      p_tilde_old = p_tilde_new;
      x_old = x_new;
    end
  end
  x = x_new;
end