function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
one = ones(m,1);
theta

z = X * theta ;
y;
sm = sigmoid(z);
left = -y .* log(sm);
r = (1 - y) .* log(1  - sm);

%J = 1/m * ((left - r)'  * one);

theta_one = theta(2:length(theta),1)
thetaSum = lambda/(2*m) * ((theta_one .* theta_one)' * ones(length(theta_one),1))

%grad = 1/m * sum ( X' * (sm - y)) + lambda/m * ( ones(length(theta_one),1)'  * theta_one));

size(X)
X(1,:)
X(2,:)
pause;
for i = 1:length(theta )
  if(i == 1)
    grad(i) =( 1/m) * sum ( X(:,i)' * (sm - y)); 
  else
    grad(i) = (1/m) * sum ( X(:,i)' * (sm - y)) + (lambda/m) * theta(i);


end


J = 1/m * ((left - r)'  * one) + thetaSum;

%theta = theta - grad;





% =============================================================

end
