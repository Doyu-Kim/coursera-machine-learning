function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); %the number of clusters

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); %the number of training example

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
nm = zeros(1,K);

for i=1:size(X,1)
  
%
%test1 : 0.08 sec
%
  
  %t = repmat(X(i,:),K,1).-centroids;
  
  %temp = t*t'; % 3 x 3 dimension
  
  %nm = diag(temp);
  
  %[val,idx(i)] = min(nm);
  
%
%test2 : 0.02 sec
%
  
  for j=1:K
    
    t = X(i,:).-centroids(j,:);
    
    nm(j) = t*t'; % 1 x 1 dimension
    
  end

  [val,idx(i)] = min(nm);
  
end

% =============================================================

end

