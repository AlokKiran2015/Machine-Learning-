function [w1 w2 w trainerror Testerror] = mlDropout(X,Y,TestX,TestY,eta,hlone,hltwo,nEpochs,batch_size)
tic
%w1 w2 w trainerror
N=size(X,1);  % 21999
D=size(X,2);  %  1024

K=size(Y,2);  % 1


winitial = -0.01+(0.02)*rand(D,hlone);% weight between input layer and first hidden layer 1024x512
w1=[zeros(1,hlone);winitial];  %1025x512

winitial2 = -0.01+(0.02)*rand(hlone,hltwo);  % wights b  512x64
w2=[zeros(1,hltwo);winitial2]; %513x64
wi = -0.01+(0.02)*rand(hltwo,K);
w=[zeros(1,K);wi]; %65x1
trainerror=zeros(nEpochs,1);
Testerror=zeros(nEpochs,1);


Drop_Prob=0.5;
remainder=rem(N,batch_size);
for i=1:nEpochs
    error = 0;
    for j=1:batch_size:N-remainder
        
%     bt=X(iporder(j),:);
%     t_label=Y(iporder(n),:);
%+biasMat(j:j+batch_size-1,:)
x1=X(j:j+batch_size-1,:);
%disp(size(x1));
%size(w1)
x=([ones(batch_size,1) x1])*w1;
%disp(size(x));  
h1=sigmoid(x);  %100x512  input for hidden layer two
 

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

d1=([ones(batch_size,1) h2]')*delta1;
ss=randperm(size(d1,2));
pp=int16(Drop_Prob*size(d1,2));
delta2=delta1*(w(2:end,:)').*(h2.*(1-h2));
for oo=1:pp
d1(:,ss(oo))=0;  %  dropout for last layer
 delta2(:,ss(oo))=0;   
end
 wnew = w-(eta/batch_size)*d1; %% output layer to second hidden layer
% disp(size(wnew));
w=wnew;
%disp(w)    
%%

%disp(size(d2));
d2=(([ones(batch_size,1) h1]')*delta2);
s2=randperm(size(d2,2));
p1=int16(Drop_Prob*size(d2,2));
 delta3=(delta2*(w2(2:end,:)')).*h1.*(1-h1);
for k=1:p1
    
   d2(:,s2(k))=0; 
    delta3(:,s2(k))=0;
end


wnew2=w2-(eta/batch_size)*d2;
w2=wnew2;
%disp('second weight');
%disp(size(w2));
 %%  hidden layer one to input layer 

 %disp(size(d3));
 d3=(([ones(batch_size,1) x1]')*delta3);
 
 s3=randperm(size(d3,2));    
 p2=int16(Drop_Prob*size(d3,2));
 
 for g=1:p2
     
    d3(:,s3(g))=0;  % drop out for hidden layer one to input layer
     
 end
 
 
 wnew3=w1-(eta/batch_size)*d3;
 w1=wnew3;
 %disp(size(w1));

    end
    
    ydashTest=mltest(TestX,w1,w2,w);
     ert = (ydashTest-TestY);
     ert = .5.*ert.*ert;
     ert =sum(ert)/size(TestY,1);
    er=error/N;
    trainerror(i)=er;
disp(sprintf('training error after epoch %d: %f\n',i,trainerror(i)));

 
    Testerror(i)=ert;
    disp(sprintf('test error after epoch %d: %f\n',i,Testerror(i)));
end

end