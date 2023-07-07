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

Xss = [];
for i=1:K
    Xss = [Xss; Xs{i}];
end

Xss = Xss';
Xtt = Xss;

for i=1:K
    Xs{i} = Xs{i}';
end
 

%% training data of two views
Ys = Compound';


%% Stack to Achieve Big Matrix Xs and Xt
s = 0;
t = 0;
for i=1:K
    s = s + size(Xs{i}, 1);
    t = t + size(Xs{i}, 2);
end
n = size(Xs{1}, 2);

d = 200;

%% call low-rank common subspace function
P = LRCS(Xtt,Xss,t,s,n,K,d);

%% Calculate the recognition rate
Zs = P'*Xs;
Zt = P'*Xt;
Cls = cvKnn(Zt,Zs,Ys,1);
acc = length(find(Cls==Yt))/length(Yt);
fprintf('Results+NN=%0.4f\n',acc);

