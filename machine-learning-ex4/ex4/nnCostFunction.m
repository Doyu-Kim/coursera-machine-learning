function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%
%Part 1 : Compute Cost Function with Forward Propagation
%

X = [ones(m,1) X];

J = 0;

%compareNum = [1:num_labels];

%
%Y Transform to 5000 x 10
%

%yTrans = zeros(m,num_labels); %5000 x 10

yTrans = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

%for cnt=1:m%1 to 5000
    
  %idx = find(compareNum==y(cnt)); %if result == false, what returns? null? or zero? or what?
    
  %yTrans(cnt,idx) = 1;
     
  %end 

%end

%yd = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

%yVec = zeros(m,num_labels);

%for i=1:m
  
  %yVec(i,y(i))=1;
  
%end  

%disp(yTrans);

%
%Comput Cost Function J
%

%for i=1:m
  
  %a2 = sigmoid(X(i,:)*Theta1');

  %m2 = size(a2,1);

  %a2 = [ones(m2,1) a2];

  %a3 = sigmoid(a2*Theta2'); %h_theta(X)
  
  %tempSum = 0;
  
  %for j=1:length(a3)
    
    %temp = [yTrans(i,j)*log(a3(j))]+[(1-yTrans(i,j))*log(1-a3(j))];
    
    %tempSum = tempSum + temp;
    
  %end

  %J = J+tempSum;  
  
%end  

%J = (-1/m)*J;

z2 = X*Theta1';

a2 = sigmoid(z2);

a2 = [ones(m,1) a2];

z3 = a2*Theta2';

a3 = sigmoid(z3);

J = (-1/m)*[sum(sum([yTrans.*log(a3)+(1-yTrans).*log(1-a3)]))]; % 1x1

%
%2nd Trial : Fail
%

%jSum = 0;

%iterNum = size(a3,2);

%for i=1:m
  
  %tempSum = 0;
  
  %for j=1:iterNum
  
    %temp = y(i)*log(a3(i,j))+(1-y(i))*log(1-a3(i,j));
    
    %tempSum = tempSum + temp;
  
  %end

  %jSum = jSum+tempSum;

%end

%J = (-1/m)*jSum;

%
%1st Trial : Fail
%

%for iter=1:size(a3,2)
  
  %temp = y'*log(a3(:,iter))+(1-y')*log(1-a3(:,iter));
  
  %tempSum = tempSum + temp;
  
%end

%J = (-1/m)*tempSum;


%
%Part 2 : Regulazied Cost Function
%

%lenth1 = size(Theta1,2);

%lenth2 = size(Theta2,2);

%theta1Sum = sum(Theta1(:,2:end).^2);

%theta2Sum = sum(Theta2(:,2:end).^2);

%theta1Sum = 0;
%theta2Sum = 0;

%for i=1:size(Theta1,1)
  
  %temp = sum(Theta1(i,2:end).^2);
  
  %theta1Sum = theta1Sum+temp;
  
%end

%for j=1:size(Theta2,1)
  
  %temp = sum(Theta2(j,2:end).^2);
  
  %theta2Sum = theta2Sum+temp;
  
%end     

%reg = (lambda/(2*m))*[theta1Sum+theta2Sum];

theta1Reg = sum(sum(Theta1(:,2:end).^2));

theta2Reg = sum(sum(Theta2(:,2:end).^2));

regTheta = (lambda/(2*m))*(theta1Reg + theta2Reg);

J = J + regTheta;

%
%Part 3 : Back Propagation
%

%delta3 = zeros(size(a3,2));

capDelta1 = zeros(size(Theta1));
  
capDelta2 = zeros(size(Theta2));

for t=1:m
    
  %delta3 = a3.-yTrans(t,:); %a3 need to recompute?
  
  %delta2 = (Theta2*delta3').*(sigmoidGradient(X*Theta1'));
 
  %capDelta1 = capDelta1 + (delta2(2:end)'*X);
 
  %capDelta2 = capDelta2 + (delta3'*a2);
  
  z2 = X(t,:)*Theta1';
  
  ta2 = sigmoid(z2);

  %tm2 = size(ta2,1);

  %ta2 = [ones(tm2,1) ta2];
  ta2 = [1 ta2]; %if row is not 1, it is not working correctly.
  
  z3 = ta2*Theta2';

  ta3 = sigmoid(z3); %h_theta(X)
  
  delta3 = ta3 -yTrans(t,:);
  
  %delta2 = (delta3*Theta2(:,2:end)).*(sigmoidGradient(z2)); % size problem
  
  delta2 = (delta3*Theta2).*(ta2.*(1-ta2));
  %delta2 = (delta3*Theta2).*[1 sigmoidGradient(z2)]; %more spend time
  
  capDelta1 = capDelta1+delta2'(2:end)*X(t,:);
  
  capDelta2 = capDelta2+delta3'*ta2;
  
end

unregTheta1 = capDelta1(:,1);

unregTheta2 = capDelta2(:,1);

regTheta1 = capDelta1(:,2:end)+(lambda*Theta1(:,2:end));
  
regTheta2 = capDelta2(:,2:end)+(lambda*Theta2(:,2:end));

%Theta1_grad = (1/m).*capDelta1(:,2:end);

%Theta2_grad = (1/m).*capDelta2(:,2:end);

Theta1_grad = (1/m).*[unregTheta1 regTheta1];
Theta2_grad = (1/m).*[unregTheta2 regTheta2];


%Occurs 0 divide Warning : Not good Example
%Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
%Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

%
% regularization
%

%regTheta1 = (lambda/m).*Theta1(:,2:end);

%regTheta2 = (lambda/m).*Theta2(:,2:end);

%Theta1_grad = Theta1_grad(:,2:end)+regTheta1;

%Theta2_grad = Theta2_grad(:,2:end)+regTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
