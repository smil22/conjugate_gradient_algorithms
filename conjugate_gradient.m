function [x, iter,tab_r] = conjugate_gradient(A,b,x0,tol,itermax)
  %Conjugate gradient algorithm. A must be symetric positive-definite.
  %- x0: starting vector
  %- tol: accuracy
  %- itermax: maximum iterations number before stopping algorithm
  %- x: solution founded
  %- tab_r: residuals vector
  %- iter: last iteration before stopping or finding the solution
  tab_r = zeros(1,itermax);
  n = length(A);
  x_new = zeros(n,1);
  iter = 0;
  x_old = x0;
  r_old = b - (A * x0);
  tab_r(1) = norm(r_old);
  p_old = r_old;
  for k=0:itermax
    Ap = A * p_old;
    r_old_r_old = r_old' * r_old;
    alpha = r_old_r_old/(Ap' * p_old);
    x_new = x_old + (alpha * p_old);
    r_new = r_old - alpha * Ap;
    beta = (r_new' * r_new)/r_old_r_old;
    p_new = r_new + (beta * p_old);
    tab_r(k+2) = norm(r_new);
    if norm(r_new) < tol
      x = x_new;
      iter = k;
      tab_r = tab_r(:,[1:k+2]);
      return;
    else
      r_old = r_new;
      p_old = p_new;
      x_old = x_new;
    end
  end
  x = x_new;
  iter = itermax;
end