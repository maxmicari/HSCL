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

%A = [1, 0; 0, 0];
%B = [1,1; 0,0];

%Subject 1
[X,Y,Xmw,Ymw] = concatData(1, Data);

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
plot(X2,'r') 
hold on
plot(X3,'r') 
hold on
plot(X6,'r') 
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

%Subject 2
[Y2,X2] = chop(Data.subject1{1, 1}.trial_2{1, 1}.speed{1,1});
[Y3,X3] = chop(Data.subject1{1, 1}.trial_3{1, 1}.speed{1,1});
[Y6,X6] = chop(Data.subject1{1, 1}.trial_6{1, 1}.speed{1,1});
[Y7,X7] = chop(Data.subject1{1, 1}.trial_7{1, 1}.speed{1,1});
[Y10,X10] = chop(Data.subject1{1, 1}.trial_10{1, 1}.speed{1,1});
[Y11,X11] = chop(Data.subject1{1, 1}.trial_11{1, 1}.speed{1,1});
[Y12,X12] = chop(Data.subject1{1, 1}.trial_12{1, 1}.speed{1,1});

function [arrY,arrX] = chop(arr)
X_1 = arr.';
[~,i]=min(X_1);
X1 = X_1(:,1:i);
Y1 = X1;
arrX = X1(1:end-1);
arrY = Y1(2:end);
end

function [X,Y,Xmw,Ymw] = concatData(subject_number,Data)
S = Data.(strcat('subject', num2str(subject_number)));
%MINDWANDER NO
if subject_number == 1
    [Y2,X2] = chop(S{1, 1}.trial_2{1, 1}.speed{1,1});
    [Y3,X3] = chop(S{1, 1}.trial_3{1, 1}.speed{1,1});
    [Y6,X6] = chop(S{1, 1}.trial_6{1, 1}.speed{1,1});
    [Y7,X7] = chop(S{1, 1}.trial_7{1, 1}.speed{1,1});
    [Y10,X10] = chop(S{1, 1}.trial_10{1, 1}.speed{1,1});
    [Y11,X11] = chop(S{1, 1}.trial_11{1, 1}.speed{1,1});
    [Y12,X12] = chop(S{1, 1}.trial_12{1, 1}.speed{1,1});
    X = [X2,X3,X6,X7,X10,X11,X12];
    Y = [Y2,Y3,Y6,Y7,Y10,Y11,Y12];
%MINDWANDER YES
    [Y4,X4] = chop(S{1, 1}.trial_4{1, 1}.speed{1,1});
    [Y5,X5] = chop(S{1, 1}.trial_5{1, 1}.speed{1,1});
    [Y8,X8] = chop(S{1, 1}.trial_8{1, 1}.speed{1,1});
    [Y9,X9] = chop(S{1, 1}.trial_9{1, 1}.speed{1,1});
    Xmw = [X4,X5,X8,X9];
    Ymw = [Y4,Y5,Y8,Y9];
end

end