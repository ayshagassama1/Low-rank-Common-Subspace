%clc
%clear all
%close all
%% load 2-view data
K = 5; %% the number of view/classes/sources
Xs = cell(1,K);
si = cell(1,K);

%% data preparation
load neuroderisk-equal.mat
Xs{1} = uca';
Xs{2} = novartis';
Xs{3} = sard';
Xs{4} = unifi';
Xs{5} = msd';

G = cell(1,K);
G{1} = G_uca';
G{2} = G_novartis';
G{3} = G_sard';
G{4} = G_unifi';
G{5} = G_msd';

for i=1:K
    Xs{i} = Xs{i} .* G{i};
end

%% Test set
n = size(Xs{1}, 2);
n_test = floor(0.3 * n);
Xt = [];
rand_test = randperm(n, n_test);
train_set = setdiff(1:n, rand_test);
for i=1:K
    Xt = [Xt; Xs{i}(:, rand_test)];
    Xs{i}(:, rand_test) = [];
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
Xtt = [];
for i=1:K
    Xtt = [Xtt, Xs{i}];
end

Xtt = [Xtt; Xtt; Xtt; Xtt; Xtt];
d = 88;
n = n - n_test;
%% call low-rank common subspace function
P = LRCS(Xtt,Xss,t,s,n,K,d);

%% Labels
%Y_uca = Y_uca'; Y_novartis = Y_novartis'; Y_sard = Y_sard'; Y_unifi = Y_unifi'; Y_msd = Y_msd';
Y = [Y_uca, Y_novartis, Y_sard, Y_unifi, Y_msd];
Y = Y';
Ys = [Y(train_set); Y(train_set); Y(train_set); Y(train_set); Y(train_set)];
Yt = [Y(rand_test); Y(rand_test); Y(rand_test); Y(rand_test); Y(rand_test)];
%Yt = Yt';
%% Calculate the recognition rate
Zs = P'*Xs2;
Zt = P'*Xt;
Cls = cvKnn(Zt,Zs,Ys,1);
acc = length(find(Cls==Yt))/length(Yt);
fprintf('Results+NN=%0.4f\n',acc);

