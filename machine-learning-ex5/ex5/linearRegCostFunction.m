function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

%
%1-2. Compute Regularized linear regression cost function
%

h = X*theta; %(12x2) * (2x1) = (12x1)

J = (1/(2*m))*sum((h-y).^2);

regJ = (lambda/(2*m))*sum(theta(2:end).^2);

J = J + regJ;

%
%1-3. Compute Regularized linear regression gradient
%

%gradOne = (1/m)*sum((X(:,1)'*h)-(X(:,1)'*y)); %(1x12) * (12x1) = (1x1)

%regGrad = (1/m)*sum((X(:,2:end)'*h)-(X(:,2:end)'*y))+(lambda/m)*theta(2:end); %(1x12) * (12x1) = (1x1) 

%grad = [gradOne;regGrad]; %(2x1)

%grad = (1/m)*[(X'*h)-(X'*y)]; % (2x1)

grad = (1/m)*[X'*(h-y)];

grad(2:end) = grad(2:end)+[(lambda/m).*theta(2:end)];

%regGrad = (lambda/m).*theta;

%regGrad(1) = 0;

%grad = (1/m)*[X'*(h-y)]+ regGrad;

%regTheta = theta;

%regTheta(1) = 0;

%grad = (1/m)*[X'*(h-y)]+(lambda/m)*regTheta;

% =========================================================================

grad = grad(:);

end
