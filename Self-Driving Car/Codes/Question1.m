% This function is the primary driver for homework 3 part 1
function Question1
close all;
clear all;
clc;
% we will experiment with a simple 2d dataset to visualize the decision
% boundaries learned by a MLP. Our goal is to study the changes to the
% decision boundary and the training error with respect to the following
% parameters
% - increasing overlap between the data points of the different classes
% - increasing the number of training iterations
% - increase the number of hidden layer neurons
% - see the effect of learning rate on the convergence of the network
 
 
% centroid for the three classes
c1=[1 1];
c2=[3 1];
c3=[2 3];
 
% standard deviation for the three classes
% "increase this quantity to increase the overlap between the classes"
sd=0.2;
 
% number of data points per class
N=100;
 
rand('seed', 1);
 
% generate data points for the three classes
%x1,x2 and x3 are each Nx2 matrices
x1=randn(N,2)*sd+ones(N,1)*c1;
x2=randn(N,2)*sd+ones(N,1)*c2;
x3=randn(N,2)*sd+ones(N,1)*c3;
 
% generate the labels for the three classes in the binary notation
y1= repmat([1 0 0],N,1);
y2= repmat([0 1 0],N,1);
y3= repmat([0 0 1],N,1);
 
% creating the test data points
a1min = min([x1(:,1);x2(:,1);x3(:,1)]);
a1max = max([x1(:,1);x2(:,1);x3(:,1)]);
 
a2min = min([x1(:,2);x2(:,2);x3(:,2)]);
a2max = max([x1(:,2);x2(:,2);x3(:,2)]);
 
[a1 a2] = meshgrid(a1min:0.1:a1max, a2min:0.1:a2max);
 
testX=[a1(:) a2(:)];
 
% Experimenting with MLP
 
% number of epochs for training
nEpochs = 1000;
 
% learning rate
eta = 0.01;
 

ct=0;
id=[];
tr=[];

for i=[2,4,8,16,32,64]  % here the hidden layer units changes from 2 to 64 at rate of power of 2
 
% train the MLP using the generated sample dataset
[w, v, trainerror] = mlptrain([x1;x2;x3],[y1;y2;y3], i, eta, nEpochs);
%%
% plot the train error againt the number of epochs
% figure; plot(1:nEpochs, trainerror, 'b:', 'LineWidth', 2);  
% xlabel('Epochs');
% ylabel('Training Error');
% title('hidden layer = 2');

%% 
ydash = mlptest(testX, w, v);
 
[val,idx] = max(ydash, [], 2);
 
label = reshape(idx, size(a1));

ct=ct+1;
 id(:,ct)=idx;
 tr(ct,:)=trainerror;
 
 
% ploting the approximate decision boundary
% -------------------------------------------
 
figure;
imagesc([a1min a1max], [a2min a2max], label), hold on,
set(gca, 'ydir', 'normal'),
 
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.9 1 0.9; 0.9 0.9 1];
colormap(cmap);
 
% plot the training data
plot(x1(:,1),x1(:,2),'r.', 'LineWidth', 2),
plot(x2(:,1),x2(:,2),'g+', 'LineWidth', 2),
plot(x3(:,1),x3(:,2),'bo', 'LineWidth', 2),
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');
title(['Number of hidden layer = ' num2str(i)]);
end


%plot the train error againt the number of epochs
figure; plot(1:nEpochs, tr(1,:), 'b', 'LineWidth', 2);
hold on;
plot(1:nEpochs, tr(2,:), 'g', 'LineWidth', 2);
hold on;
plot(1:nEpochs, tr(3,:), 'r', 'LineWidth', 2);
hold on;
plot(1:nEpochs, tr(4,:), 'y', 'LineWidth', 2);
hold on;
plot(1:nEpochs, tr(5,:), 'm', 'LineWidth', 2);
hold on;
plot(1:nEpochs, tr(6,:), 'k', 'LineWidth', 2);

xlabel('Epochs');
ylabel('Training Error');
hold off;
legend('2','4','8','16','32','64');




%  
% viewing the decision surface for the three classes
% ydash1 = reshape(ydash(:,1), size(a1));
% ydash2 = reshape(ydash(:,2), size(a1));
% ydash3 = reshape(ydash(:,3), size(a1));
% 
% figure;
% surf(a1, a2, ydash1, 'FaceColor', [1 0 0], 'FaceAlpha', 0.5), hold on,...
% surf(a1, a2, ydash2, 'FaceColor', [0 1 0], 'FaceAlpha', 0.5), hold on,...
% surf(a1, a2, ydash3, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5);
 
function [w v trainerror] = mlptrain(X, Y, H, eta, nEpochs)
% X - training data of size NxD
% Y - training labels of size NxK
% H - the number of hidden units
% eta - the learning rate
% nEpochs - the number of training epochs
% define and initialize the neural network parameters
 
% number of training data points
N = size(X,1);
% number of inputs
D = size(X,2); % excluding the bias term
% number of outputs
K = size(Y,2);
 
% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% w is a Hx(D+1) matrix
w = -0.3+(0.6)*rand(H,(D+1));
 
% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% v is a Kx(H+1) matrix
v = -0.3+(0.6)*rand(K,(H+1));
 
% randomize the order in which the input data points are presented to the
% MLP
iporder = randperm(N);
%trainerror=zeros(1,nEpochs);
% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    for n = 1:N
        % the current training point is X(iporder(n), :)
        % forward pass
        % --------------
        % input to hidden layer
        % calculate the output of the hidden layer units - z
        % ---------
        %'TO DO'%
        bt=X(iporder(n),:);    %  shuffling the datapoints.
        t_label=Y(iporder(n),:);%
       
        z=sgmd(w*transpose([1 bt]));%applied sigmoid function after adding bias input to X 
       
        % ---------
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        %'TO DO'%
        z2=[1;z];  % again adding the bias terms to hidden layer 
     
       ydash=exp(v*z2)/sum(exp(v*z2));   %using softmax function as activation function
       
       % ydash is the output of output layer.
       % Forward pass completes here
       %%
        % ---------
         %%%%%%%%
        % backward pass
        % ---------------
        % update the weights for the connections between hidden and
        % outlayer units
        % ---------
        %'TO DO'%
        %cross entropy is the error function
        d1=(transpose(t_label)-ydash)*transpose([1;z]);
        V=-eta*d1;  %error term
        Vinitial=v;
        v=Vinitial-V; % update equation
        % ---------
        % update the weights for the connections between the input and
        % hidden later units
        % ---------
        %'TO DO'%
        p=transpose(Vinitial(:,2:end));
        out=transpose(t_label);
   %%((z.*(1-z))  = sgmd(input)*(1-sgmd(input)) i.e. derivative of sigmoid
   d2=p*(out-ydash)*ones(1,D+1).*((z.*(1-z))*[1 bt]);
        W=-eta*d2;  % change in weight
        w=w-W; %weight update 
        % ---------
        
        %% backward pass completes here
    end
    ydash = mlptest(X, w, v);   %ydash has the dimension NxK
    % compute the training error
    % ---------
    %'TO DO'% uncomment the next line after adding the necessary code
    lg=log(ydash);
    trainerror(epoch) =-sum(sum(Y.*lg));
    %% Cross entropy function to calculate training error
    % ---------
    disp(sprintf('training error after epoch %d: %f\n',epoch,...
        trainerror(epoch)));
end
return;
 
function ydash = mlptest(X, w, v)
% forward pass of the network
 
% number of inputs
N = size(X,1);
 
% number of outputs
K = size(v,1);
ydash=zeros(N,K);
for i=1:N
% forward pass to estimate the outputs
% --------------------------------------
% input to hidden for all the data points
% calculate the output of the hidden layer units
% ---------
%'TO DO'%
Y=X(i,:);
Y=transpose(Y);
Y=[1;Y];  %adding bias
z=sgmd(w*Y);  %calling sgmd i.e. sigmoid function
% ---------% hidden to output for all the data points
% calculate the output of the output layer units
% ---------
%'TO DO'%
zo=[1;z];  % output of hidden layer + bias
finaloutput=exp(v*zo)/sum(exp(v*zo));%softmax function
finaloutput=transpose(finaloutput);
ydash(i,:)=finaloutput;  % output of output layer.
end

return;

