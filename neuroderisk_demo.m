clc
clear all
close all
%% load 2-view data
K = 5; %% the number of view/classes/sources
Xs = cell(1,K);
si = cell(1,K);

%% data preparation
load neuroderisk-equal.mat
Xs{1} = uca;
Xs{2} = novartis;
Xs{3} = sard;
Xs{4} = unifi;
Xs{5} = msd;

G = cell(1,K);
G{1} = G_uca;
G{2} = G_novartis;
G{3} = G_sard;
G{4} = G_unifi;
G{5} = G_msd;

for i=1:K
    %Xs{i} = NormalizeFea(Xs{i});
    Xs{i} = Xs{i} .* G{i};
end

Xt = [];
for i=1:K
    Xt = [Xt;Xs{i}];
end
Xt = Xt';
Yt = [Yt1;Yt2];

%% training data of two views
Xs = [Xs1;Xs2]';
Ys = [Ys1;Ys2];
Xs1 = Xs1';
Xs2 = Xs2';
%% Stack to Achieve Big Matrix Xs and Xt
s1 = size(Xs1,2);
s2 = size(Xs2,2);
t2 = size(Xs2,2);
[n,t1] = size(Xs1); %% high dim
Xss = [Xs1, zeros(size(Xs2));
    zeros(size(Xs1)), Xs2];

Xtt = [Xs1, Xs2];
Xtt = [Xtt;Xtt];
s = s1+s2;
t = t1+t2;
d = 200;

%% call low-rank common subspace function
P = LRCS(Xtt,Xss,t,s,n,K,d);

%% Calculate the recognition rate
Zs = P'*Xs;
Zt = P'*Xt;
Cls = cvKnn(Zt,Zs,Ys,1);
acc = length(find(Cls==Yt))/length(Yt);
fprintf('Results+NN=%0.4f\n',acc);

