function ydash = mltest(X,w1,w2,w)

N = size(X,1); % number of input

x1=[ones(N,1) X];
x=x1*w1;

out1=sigmoid(x);  % output 1;
 
 x2=([ones(N,1) out1])*w2;
 
 out2=sigmoid(x2);
 
 
   ydash=([ones(N,1) out2])*w; %final out put
   
   
end


