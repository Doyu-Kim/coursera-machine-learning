function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 10;
sigma = 0.3;

tryList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%C, sigma example range : 0.01 ~ 30
min = 1; recC = 0; recSigma = 0;

x1 = [1 2 1]; x2 = [0 4 -1];

tryLen = length(tryList);

for i=1:tryLen
  
  C=tryList(i); 
  
  for j=1:tryLen
    
    sigma=tryList(j);
  
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

    predictions = svmPredict(model,Xval);

    predicError = mean(double(predictions ~= yval));
    %mean function : compute mean
    %double function : change data type to double type
    
    disp(sprintf("predicError = %f",predicError));
  
    if min > predicError
    
      min = predicError;
      
      recC = C;
      
      recSigma = sigma;
      
      disp(sprintf("min = %f",min));
      
      disp(sprintf("C = %f",recC));
      
      disp(sprintf("sigma = %f",recSigma));
    
    end
  
  end

end  

%c=1,sigma=0.3 >> predicError=0.075

C = recC;

sigma = recSigma;

%=========================================================================

end
