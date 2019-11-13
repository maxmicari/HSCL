%% Kernel Embeddings Example (First-Hitting Time Problem)
% Kernel embeddings example showing the first-hitting time problem
% for a double integrator system.
%
%%
% Specify the time horizon, the safe set $\mathcal{K}$, and the target set
% $\mathcal{T}$.

%FIRST First-Hitting Time Problem (70mph ---> 30mph)
N = 1000; %look at toggling this 2000+
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
%[X1,Y1,X1mw,Y1mw] = concatData(1, Data);
%Subject 3
%[X3,Y3,X3mw,Y3mw] = concatData(3,Data);
%Subject 4
[X4,Y4,X4mw,Y4mw] = concatData(4,Data);
%All
[Xno,Yno,Xya,Yya] = concatALLData(Data);
%%
% Create a sample-based stochastic system.
% sys1 = srt.systems.SampledSystem('X', X1, 'Y', Y1);
% sysMW1 = srt.systems.SampledSystem('X', X1mw, 'Y', Y1mw);
% sys3 = srt.systems.SampledSystem('X', X3, 'Y', Y3);
% sysMW3 = srt.systems.SampledSystem('X', X3mw, 'Y', Y3mw);
sys4 = srt.systems.SampledSystem('X', X4, 'Y', Y4);
sysMW4 = srt.systems.SampledSystem('X', X4mw, 'Y', Y4mw);
sysAll = srt.systems.SampledSystem('X', Xno, 'Y', Yno);
sysMWAll = srt.systems.SampledSystem('X', Xya, 'Y', Yya);
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

% results1 = SReachPoint(prb, alg, sys1, Xtest);
% resultsMW1 = SReachPoint(prb, alg, sysMW1, Xtest);
% results3 = SReachPoint(prb, alg, sys3, Xtest);
% resultsMW3 = SReachPoint(prb, alg, sysMW3, Xtest);
results4 = SReachPoint(prb, alg, sys4, Xtest);
resultsMW4 = SReachPoint(prb, alg, sysMW4, Xtest);
resultsAll = SReachPoint(prb, alg, sysAll, Xtest);
resultsMWAll = SReachPoint(prb, alg, sysMWAll, Xtest);
%%
% View the results.
%subplot(1,2,1)
% plot(results1.Pr(:,1),'r')
% hold on
% plot(resultsMW1.Pr(:,1),'r--')
% hold on
% plot(results3.Pr(:,1),'b')
% hold on
% plot(resultsMW3.Pr(:,1),'b--')
% hold on
plot(results4.Pr(:,1),'g')
hold on
plot(resultsMW4.Pr(:,1),'g--')
hold on
plot(resultsAll.Pr(:,1),'b')
hold on
plot(resultsMWAll.Pr(:,1),'b--')

xlabel('N steps')
ylabel('Likelihood from 70mph')
legend('mw no sub4','mw yes sub4','mw no all subjects','mw yes all subjects') %'MW no','MW yes'
title('RKHS Results Human Driving Sim (subj 4 v All)')
hold off

% subplot(1,2,2)
% plot(X2,'r') 
% hold on
% plot(X3,'r') 
% hold on
% plot(X6,'r') 
% hold on
% plot(X7,'r') 
% hold on
% plot(X10,'r') 
% hold on
% plot(X11,'r') 
% hold on
% plot(X12,'r') 
% hold on
% plot(X4,'b')
% hold on
% plot(X5,'b')
% hold on
% plot(X8,'b')
% hold on
% plot(X9,'b')
% hold on
% xlabel('time')
% ylabel('velocity')
% title('Velocity Time Graph for all trials (subj 1)')
% legend('mw no')
% hold off
%surf(X2, X2, reshape(results.Pr(1, :), 100, 100), 'EdgeColor', 'none');


function [arrY,arrX] = chop(arr)
X_1 = arr.';
[~,i]=min(X_1);
X1 = X_1(:,1:i);
Y1 = X1;
arrX = X1(1:end-1);
arrY = Y1(2:end);
end

function [Xno,Yno,Xya,Yya] = concatALLData(Data)
MindWandering = load('C:\Users\mbucci\Documents\HSCL\SReachTools-new_structure\tbx\doc\examples\MindWandering.mat');
Xno = linspace(0,0,0);
Yno = linspace(0,0,0);
Xya = linspace(0,0,0);
Yya = linspace(0,0,0);
k=1;
for i=4:16 %VarName
    for j=1:12 %trials
        V = MindWandering.MindWandering.(strcat('VarName', num2str(i)));
        if V(j) == 'NaN'
           continue 
        end
        if V(j) == 0
            disp(strcat('subject', num2str(k)))
            disp(strcat('trial_', num2str(j)))
            S = Data.(strcat('subject', num2str(k)));
            T = S{1,1}.(strcat('trial_', num2str(j)));
            [Y,X] = chop(T{1, 1}.speed{1,1});
            Xno = [Xno,X];
            Yno = [Yno,Y];
        end
        if V(j) == 1
            disp(strcat('subject', num2str(k)))
            disp(strcat('trial_', num2str(j)))
            S = Data.(strcat('subject', num2str(k)));
            T = S{1,1}.(strcat('trial_', num2str(j)));
            [Ymw,Xmw] = chop(T{1, 1}.speed{1,1});
            Xya = [Xya,Xmw];
            Yya = [Yya,Ymw];
        end
    end
    k=k+1;
plot(Xya)
plot(Xno)
end
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
if subject_number == 3
    %Subject 3
    %MW yes
    [Y4,X4] = chop(S{1, 1}.trial_4{1, 1}.speed{1,1});
    [Y10,X10] = chop(S{1, 1}.trial_10{1, 1}.speed{1,1});
    [Y11,X11] = chop(S{1, 1}.trial_11{1, 1}.speed{1,1});
    [Y12,X12] = chop(S{1, 1}.trial_12{1, 1}.speed{1,1});
    Xmw = [X4,X10,X11,X12];
    Ymw = [Y4,Y10,Y11,Y12];
    %MW no
    [Y2,X2] = chop(S{1, 1}.trial_2{1, 1}.speed{1,1});
    [Y3,X3] = chop(S{1, 1}.trial_3{1, 1}.speed{1,1});
    [Y5,X5] = chop(S{1, 1}.trial_5{1, 1}.speed{1,1});
    [Y6,X6] = chop(S{1, 1}.trial_6{1, 1}.speed{1,1});
    [Y7,X7] = chop(S{1, 1}.trial_7{1, 1}.speed{1,1});
    [Y8,X8] = chop(S{1, 1}.trial_8{1, 1}.speed{1,1});
    [Y9,X9] = chop(S{1, 1}.trial_9{1, 1}.speed{1,1});
    X = [X2,X3,X5,X6,X7,X8,X9];
    Y = [Y2,Y3,Y5,Y6,Y7,Y8,Y9];
end
if subject_number == 4
    %Subject 3
    %MW yes
    [Y3,X3] = chop(S{1, 1}.trial_3{1, 1}.speed{1,1});
    [Y4,X4] = chop(S{1, 1}.trial_4{1, 1}.speed{1,1});
    [Y7,X7] = chop(S{1, 1}.trial_7{1, 1}.speed{1,1});
    [Y10,X10] = chop(S{1, 1}.trial_10{1, 1}.speed{1,1});
    [Y11,X11] = chop(S{1, 1}.trial_11{1, 1}.speed{1,1});
    Xmw = [X3,X4,X7,X10,X11];
    Ymw = [Y3,Y4,Y7,Y10,Y11];
    %MW no
    [Y2,X2] = chop(S{1, 1}.trial_2{1, 1}.speed{1,1});
    [Y5,X5] = chop(S{1, 1}.trial_5{1, 1}.speed{1,1});
    [Y6,X6] = chop(S{1, 1}.trial_6{1, 1}.speed{1,1});
    [Y8,X8] = chop(S{1, 1}.trial_8{1, 1}.speed{1,1});
    [Y9,X9] = chop(S{1, 1}.trial_9{1, 1}.speed{1,1});
    [Y12,X12] = chop(S{1, 1}.trial_12{1, 1}.speed{1,1});
    X = [X2,X5,X6,X8,X9,X12];
    Y = [Y2,Y5,Y6,Y8,Y9,Y12];
end
end