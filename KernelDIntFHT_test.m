%% Kernel Embeddings Example (First-Hitting Time Problem)
% Kernel embeddings example showing the first-hitting time problem
% for a double integrator system.
%
%%
% Specify the time horizon, the safe set $\mathcal{K}$, and the target set
% $\mathcal{T}$.

%FIRST First-Hitting Time Problem (70mph ---> 30mph)
N = 500; %look at toggling this 2000+
K = srt.Tube(N, Polyhedron('lb', [10], 'ub', [100])); %Safe space 10-100
T = srt.Tube(N, Polyhedron('lb', [25], 'ub', [35])); %Target space 25-45

prb = srt.problems.FirstHitting('ConstraintTube', K, 'TargetTube', T);

%% System Definition
% Generate input/output samples for a double integrator system.
%
% $$x_{k+1} = A x_{k} + w_{k}, \quad w_{k} \sim \mathcal{N}(0, 0.01 I)$$
%

%MINDWANDER?

%Chop Function

%X_raw = [Data.subject1{1, 1}.trial_1{1, 1}.speed{1,1}.', Data.subject1{1, 1}.trial_2{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_3{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_4{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_5{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_6{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_7{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_9{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_10{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_11{1, 1}.speed{1,1}.',Data.subject1{1, 1}.trial_12{1, 1}.speed{1,1}.'];
%X_2 = Data.subject1{1, 1}.trial_2{1, 1}.speed{1,1}.';
%,Data.subject1{1, 1}.trial_3{1, 1}.speed{1,1}.'];
%Data.subject1{1, 1}.trial_8{1, 1}.speed{1,1}.',
%X = [X_1(1:end-1),X_2(1:end-1)];
%U_acc = Data.subject1{1, 1}.trial_1{1, 2}.accelerate{1,1}.';
%U_acc = U_acc(1:end-1);
%U_brake = Data.subject1{1, 1}.trial_1{1, 3}.brake{1,1}.';
%U_brake = U_brake(1:end-1);

%A = [1, 0; 0, 0];
%B = [1,1; 0,0];

X_1 = Data.subject1{1, 1}.trial_1{1, 1}.speed{1,1}.';
[~,i]=min(X_1);
X_2 = Data.subject1{1, 1}.trial_2{1, 1}.speed{1,1}.';
[~,j]=min(X_2);
X_3 = Data.subject1{1, 1}.trial_3{1, 1}.speed{1,1}.';
[~,k]=min(X_3);

X1 = X_1(:,1:i);
Y1 = X1;
X1 = X1(1:end-1);
Y1 = Y1(2:end);

X2 = X_2(:,1:j);
Y2 = X2;
X2 = X2(1:end-1);
Y2 = Y2(2:end);

X3 = X_3(:,1:k);
Y3 = X3;
X3 = X3(1:end-1);
Y3 = Y3(2:end);

X = [X1,X2,X3];
Y = [Y1,Y2,Y3];
%Y = X + U_acc + U_brake;

%MINDWANDER YES (subj 7-12)

%%
% Create a sample-based stochastic system.

sys = srt.systems.SampledSystem('X', X, 'Y', Y);

%% Algorithm
% Initialize the algorithm.


alg = srt.algorithms.KernelEmbeddings('sigma', 0.1, 'lambda', 1);

%%
% Call the algorithm on a second trial

%X2 = Data.subject1{1, 1}.trial_8{1, 1}.speed{1,1}'; %use trial 8 bc voided in map
Xtest = [70]; %use this for 
%s=linspace(-1, 1, 100);
%X2 = sampleunif(s2, s2);
%U2 = zeros(1, size(X2, 2));
%U2 = [36];

results = SReachPoint(prb, alg, sys, Xtest);   %, U2); %U2, for 70-->30 direct

%%
% View the results.
plot(results.Pr(:,1))
xlabel('N steps')
ylabel('Likelihood from 70mph')
%surf(X2, X2, reshape(results.Pr(1, :), 100, 100), 'EdgeColor', 'none');
