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
T = srt.Tube(N, Polyhedron('lb', [25], 'ub', [45])); %Target space 25-45

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
%MINDWANDER NO
X_1 = Data.subject1{1, 1}.trial_2{1, 1}.speed{1,1}.';
[~,i]=min(X_1);
X_2 = Data.subject1{1, 1}.trial_3{1, 1}.speed{1,1}.';
[~,j]=min(X_2);
X_3 = Data.subject1{1, 1}.trial_6{1, 1}.speed{1,1}.';
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
X_7 = Data.subject1{1, 1}.trial_7{1, 1}.speed{1,1}.';
[~,i]=min(X_7);
X_10 = Data.subject1{1, 1}.trial_10{1, 1}.speed{1,1}.';
[~,j]=min(X_10);
X_11 = Data.subject1{1, 1}.trial_11{1, 1}.speed{1,1}.';
[~,k]=min(X_11);
X_12 = Data.subject1{1, 1}.trial_12{1, 1}.speed{1,1}.';
[~,l]=min(X_12);
X7 = X_7(:,1:i);
Y7 = X7;
X7 = X7(1:end-1);
Y7 = Y7(2:end);
X10 = X_10(:,1:j);
Y10 = X10;
X10 = X10(1:end-1);
Y10 = Y10(2:end);
X11 = X_11(:,1:k);
Y11 = X11;
X11 = X11(1:end-1);
Y11 = Y11(2:end);
X12 = X_12(:,1:l);
Y12 = X12;
X12 = X12(1:end-1);
Y12 = Y12(2:end);
X = [X1,X2,X3,X7,X10,X11,X12];
Y = [Y1,Y2,Y3,Y7,Y10,Y11,Y12];
%Y = X + U_acc + U_brake;

%MINDWANDER YES
X_4 = Data.subject1{1, 1}.trial_4{1, 1}.speed{1,1}.';
[~,imw]=min(X_4);
X_5 = Data.subject1{1, 1}.trial_5{1, 1}.speed{1,1}.';
[~,jmw]=min(X_5);
X_8 = Data.subject1{1, 1}.trial_8{1, 1}.speed{1,1}.';
[~,kmw]=min(X_8);
X_9 = Data.subject1{1, 1}.trial_9{1, 1}.speed{1,1}.';
[~,lmw]=min(X_9);
X4 = X_4(:,1:imw);
Y4 = X4;
X4 = X4(1:end-1);
Y4 = Y4(2:end);
X5 = X_5(:,1:jmw);
Y5 = X5;
X5 = X5(1:end-1);
Y5 = Y5(2:end);
X8 = X_8(:,1:kmw);
Y8 = X8;
X8 = X8(1:end-1);
Y8 = Y8(2:end);
X9 = X_9(:,1:lmw);
Y9 = X9;
X9 = X9(1:end-1);
Y9 = Y9(2:end);
Xmw = [X4,X5,X8,X9];
Ymw = [Y4,Y5,Y8,Y9];
%%
% Create a sample-based stochastic system.

sys = srt.systems.SampledSystem('X', X, 'Y', Y);
sysMW = srt.systems.SampledSystem('X', Xmw, 'Y', Ymw);

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

results = SReachPoint(prb, alg, sys, Xtest);
resultsMW = SReachPoint(prb, alg, sysMW, Xtest);
%%
% View the results.
subplot(1,2,1)
plot(results.Pr(:,1),'r')
hold on
plot(resultsMW.Pr(:,1),'b')

xlabel('N steps')
ylabel('Likelihood from 70mph')
legend('mw no','mw yes') %'MW no','MW yes'
title('RKHS Results Human Driving Sim (subj 1)')
hold off

subplot(1,2,2)
plot(X1,'r') 
hold on
plot(X2,'r') 
hold on
plot(X3,'r') 
hold on
plot(X7,'r') 
hold on
plot(X10,'r') 
hold on
plot(X11,'r') 
hold on
plot(X12,'r') 
hold on
plot(X4,'b')
hold on
plot(X5,'b')
hold on
plot(X8,'b')
hold on
plot(X9,'b')
hold on
xlabel('time')
ylabel('velocity')
title('Velocity Time Graph for all trials (subj 1)')
legend('mw no')
hold off
%surf(X2, X2, reshape(results.Pr(1, :), 100, 100), 'EdgeColor', 'none');
