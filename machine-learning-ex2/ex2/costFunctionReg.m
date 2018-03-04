function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); %gradient

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Cost Function
predictions = sigmoid(X*theta);

temp = y'*log(predictions)+(1-y')*log(1-predictions);

len = length(theta);

regExp = (lambda/(2*m))*sum(theta(2:len).^2);

J = -(1/m)*sum(temp)+regExp;

%repmat function is copy of Matrix. ex) B = repmat(A,m,n) : m by n Matrix copy of A.
%each element of B is matrix A. ex) repmat(A,2,2) : b11 = A, b12 = A, b21 = A, b22 = A
%if A size is m x n, B size is p x q. then p = (mx2), q = (nx2)

%Gradient of Cost
%gradOne = (1/m).*[X'(1,:)*sigmoid(X(:,1)*theta(1))]-[X'(1,:)*y]; %if theta = 1

%gradAll = (1/m).*[X'(2:len,:)*sigmoid(X(:,2:len)*theta(2:len))]-[X'(2:len,:)*y]-[(lambda/m)*theta(2:len)]; %if theta = not 1

%grad = [gradOne;gradAll];

h = sigmoid(X*theta);

grad(1) = (1/m).*[sum(X'(1,:)*h-X'(1,:)*y)];

for iter=2:len
  
  %grad(iter) = [(1/m).*[X'(iter,:)*sigmoid(X(:,iter)*theta(iter))]-[X'(iter,:)*y]]+[(lambda/m)*theta(iter)];
  
  grad(iter) = (1/m).*[sum(X'(iter,:)*h-X'(iter,:)*y)+lambda.*theta(iter)];

end

% =============================================================

end
