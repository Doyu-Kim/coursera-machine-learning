function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    predictions = X*theta;
    
    sqrErrors = X'*(predictions-y);
    
    delta = (1/m)*sqrErrors;
    
    theta = theta - alpha * delta;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    %plot(theta(2),J_history(iter));
    
    %disp(J_history(iter))

end

disp(sprintf('J value : %f',J_history(num_iters)));
disp(sprintf('theta 1: %f, theta 2: %f',theta(1),theta(2)));

end


