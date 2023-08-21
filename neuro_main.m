%clc
%clear all
%close all
%% load 2-view data
K = 5; %% the number of view/classes/sources
Xs = cell(1,K);
si = cell(1,K);

%% data preparation
load neuroderisk-Li.mat
Xs{1} = X1;
Xs{2} = X2;
Xs{3} = X3;
Xs{4} = X4;
Xs{5} = X5;

%% Test set
n = size(Xs{1}, 1);
n_test = floor(0.3 * n);
Xt = [];
rand_test = randperm(n, n_test);
train_set = setdiff(1:n, rand_test);
for i=1:K
    Xt = [Xt; Xs{i}( rand_test, :)];
    %Xs{i}(rand_test, :) = [];
end
Xt = Xt';

Xs2 = [];
for i=1:K
    Xs2 = [Xs2; Xs{i}];
end
Xs2 = Xs2';


for i=1:K
    Xs{i} = Xs{i}';
end



s = 0;
for i=1:K
    s = s + size(Xs{i},2);
end
t = s;


Xss = blkdiag(Xs{1}, Xs{2}, Xs{3}, Xs{4}, Xs{5});
L = blkdiag(L1(train_set, train_set), L2(train_set, train_set), L3(train_set, train_set), L4(train_set, train_set), L5(train_set, train_set));
Xtt = [];
for i=1:K
    Xtt = [Xtt, Xs{i}];
end

Xtt = [Xtt; Xtt; Xtt; Xtt; Xtt];
d = 23;
% n = n - n_test;
n = size(Xs{1},1);
%% call low-rank common subspace function
P = LRCS(Xtt,Xss,t,s,n,K,d, Lf);

%% Labels
%Y_uca = Y_uca'; Y_novartis = Y_novartis'; Y_sard = Y_sard'; Y_unifi = Y_unifi'; Y_msd = Y_msd';
Y = compound;
Y = Y';
Ys = [Y(train_set); Y(train_set); Y(train_set); Y(train_set); Y(train_set)];
Yt = [Y(rand_test); Y(rand_test); Y(rand_test); Y(rand_test); Y(rand_test)];
%Yt = Yt';
%% Calculate the recognition rate
Zs = P'*Xs2;
Zt = P'*Xt;
save('neuro-P', 'P');
save('neuro-Zs', 'Zs');
save('neuro-Zt', 'Zt');
save('neuro-Ys', 'Y');
save('neuro-Yt', 'Yt');
%Cls = cvKnn(Zt,Zs,Ys,1);
%acc = length(find(Cls==Yt))/length(Yt);
%fprintf('Results+NN=%0.4f\n',acc);