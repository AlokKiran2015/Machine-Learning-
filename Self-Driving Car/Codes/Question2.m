function [w1 w2 w trainerror Testerror] = Question2(X,Y,TestX,TestY,eta,hlone,hltwo,nEpochs,batch_size)
tic
%w1 w2 w trainerror
N=size(X,1);  % 21999
D=size(X,2);  %  1024

K=size(Y,2);  % 1x1


winitial = -0.01+(0.02)*rand(D,hlone);% weight between input layer and first hidden layer 1024x512
w1=[zeros(1,hlone);winitial];  %1025x512

winitial2 = -0.01+(0.02)*rand(hlone,hltwo);  % weights   512x64
w2=[zeros(1,hltwo);winitial2]; %513x64
wi = -0.01+(0.02)*rand(hltwo,K);
w=[zeros(1,K);wi]; %65x1
trainerror=zeros(nEpochs,1);
Testerror=zeros(nEpochs,1);
remainder=rem(N,batch_size);
for i=1:nEpochs
    error = 0;
    for j=1:batch_size:N-remainder  
        
x1=X(j:j+batch_size-1,:);
%disp(size(x1));
%size(w1)
x=([ones(batch_size,1) x1])*w1;
%disp(size(x));  
h1=sigmoid(x);  %100x512  input for hidden layer two
%  
 
 %z2=[biasMat h1];  %513x64
z2=([ones(batch_size,1) h1])*w2;
 h2=sigmoid(z2);  % output of hidden layer 2 ;; 100x64

 h3= ([ones(batch_size,1) h2])*w;           %100x1
 
 out=h3;  % 100x1  
 %disp(out);
 y=Y(j:j+batch_size-1,:);   %% forward pass completes here
 
% Finding error
   error=error + (1/2)*sum((out-y).^2);  % sum of squarred error.
 
 % go to backward pass.
delta1=(out-y);


 wnew = w-(eta/batch_size)*([ones(batch_size,1) h2]')*delta1; %% output layer to second hidden layer
% disp(size(wnew));
w=wnew;
%disp(w)  %% propagates the  
%%
delta2=delta1*(w(2:end,:)').*(h2.*(1-h2));

d2=(([ones(batch_size,1) h1]')*delta2);  % after adding bias term
%disp(size(d2));
wnew2=w2-(eta/batch_size)*d2;
w2=wnew2;
%disp('second weight');
%disp(size(w2));
 %%  hidden layer one to input layer 
 d3=([ones(batch_size,1) x1]')*(delta2*(w2(2:end,:)')).*h1.*(1-h1);

 
 wnew3=w1-(eta/batch_size)*d3;
 w1=wnew3;
 %disp(size(w1));

    end
    
    ydashTest=mltest(TestX,w1,w2,w);  % output for validation set
     ert = (ydashTest-TestY);
     ert = .5.*ert.*ert;   % mean square error of validation set
     ert =sum(ert)/size(TestY,1);
    er=error/N;  % mean square error.
    trainerror(i)=er;
    disp(sprintf('training error after epoch %d: %f\n',i,trainerror(i)));

 
    Testerror(i)=ert;  % mse correspods to i^th epoch
    disp(sprintf('test error after epoch %d: %f\n',i,Testerror(i)));  % test error corresponds to validation error
end

end