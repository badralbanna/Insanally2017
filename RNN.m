% Code that generates the data relevant to Extended Fig 15 %
% RNN trained by a partially targetted application of fullFORCE %

learn = 1;  % start at 1 for training, toggle to 0 to test

nRunTot = 10; % number of runs
N  = 1000; % network size
g = 1.2;   % scale for target generator  
g2 = 0.5;   % scale for Do-er network
T = 300; % length of trial
dt = 0.1; % integration time constant
tau = 1; % scale of dynamics
t = 0:dt:T;  %time steps
nT = 250; % number of targetted cells, 100*nT/N = percent of R cells eventually
nIn = 1200; % units getting input (AC), 100*nIn/N = percent of driven cells

input1 = zeros(N, length(t)); % pulse input, fin(t), facsimile for tones
input5 = zeros(N, length(t)); % output nosepoke, fOut(t)
inputN = zeros(N, length(t)); % noise input, onto 75% NNR, only sosetimes
freq = log([0.5 1 2 4 8 16 32]); % auditory inputs from expt, log scale
freqVec = datasample(freq, nRunTot); % one sample from above for each trial

% Selecting cells for silencing, uncomment after training for 15c,d 
% nKill = 2;
% killList = randperm(N);
% cL = killList(1:nKill);

if (learn)
    J2 = g2*randn(N,N)/sqrt(N); % initialize do-er matrix, plastic
    w = randn(N, 1)/sqrt(N); % readout weights, plastic
    uIn1 = 2*rand(N, 1) - 1; % input weights, unmodified
    uIn1(nIn+1:N) = 0; % inputs to only a few neurons
    uOut = 2*rand(N, 1) - 1; % weights for target-generator's extra input
    J1 = g*randn(N,N)/sqrt(N); % target generator matrix, unmodified
    wNR = randn(N-nT, 1)/sqrt(N-nT); % NNR readout weights for decoder
    wR = randn(nT, 1)/sqrt(nT); % R readout weights for decoder
    
    noiseLevel = 0.0;  % noise amplitude, set to 0.5 for noisy sims 
    sigN = noiseLevel*sqrt(tau/dt); % scaling to right units
end

H1 = randn(N,1);
R1 = zeros(N, length(t));
R2 = zeros(N, length(t));
Tar = zeros(N, length(t));
H2 = randn(N,1);
zt = zeros(1, length(t)); % output of network, z(t)
ztR = zeros(1, length(t));
ztNR = zeros(1, length(t));

% initialize P matrix in fullFORCE: tracks cross correlations and stability
P0 = 1;
PJ = P0*eye(nT, nT);
Pw0 = 1;
PW = Pw0*eye(N, N);
Pw0R = 1;
PWR = Pw0R*eye(nT, nT);
Pw0NR = 1;
PWNR = Pw0NR*eye(N-nT, N-nT);

tLearn = 1;
durVec = zeros(1, nRunTot);
ampVec = zeros(1, nRunTot);
OutInt = zeros(1, nRunTot);
OutIntR = zeros(1, nRunTot);
OutIntNR = zeros(1, nRunTot);
% squared error terms
% chi1 = zeros(1, nRunTot);
% chi2 = zeros(1, nRunTot);

outStart = 2000; % where you start to answer, 200ms

for nRun = 1:nRunTot %loop over trials
    nRun
    fIn1 = zeros(1, length(t));
    ampIn = freqVec(nRun);
    dur = 1000;
    fIn1(1000:1000+dur) = ampIn;
    
    fOut = zeros(1, length(t));
    if (ampIn == log(4)) % Target tone input
        fOut(1000+dur:1500+dur) = sin(pi*(0:500)./500); % make fOut(t) as bump
    else
        fOut(1000+dur:1500+dur) = 0; % make fOut(t) as no-response for off target
    end
    
    input1 = uIn1*fIn1; % tone input
    input5 = uOut*fOut; % extra input to target generator
    inputN = sigN .* randn(N, length(t)); % noise to NNR units in do-er
    inputN(1:nT, :) = 0; % setting noisy input weights to 25% of R cells to 0
    
    for tt = 1:length(t) %loop over time
        tLearn = tLearn + dt;
        R1(:, tt) = tanh(H1); % phi(x) for target generator
        R2(:, tt) = tanh(H2); % phi(x) for do-er
        zt(tt) = w'*R2(:, tt); % z(t) = \sum_jw_jR_j(t)
        ztNR(tt) = wNR'*R2(nT+1: N, tt);  % decoder for NNR cells
        ztR(tt) = wR'*R2(1:nT, tt); % decoder for R cells
        JR1 = J1*R1(:, tt) + input1(:, tt) + input5(:,tt); %J\phi term for target generator
        Tar(:, tt) = J1*R1(:, tt) + input5(:, tt); % targets
        JR2 = J2*R2(:, tt) + input1(:, tt) + inputN(:, tt); %J\phi term for Do-er
        H1 = (-H1 + JR1)*dt/tau + H1; %integration of network eqns
        H2 = (-H2 + JR2)*dt/tau + H2; %integration of network eqns
%         H2(cL, :) = 0; % setting currents to 0 for silencing expts
%                 H2(nT+1:50:N, :) = 0;
        if ((learn)&&(tLearn>0.89)) % RLS algo for training J2 and w
            tLearn = 0;
            err1 = zt(tt) - fOut(tt);
            err2 = J2(1:nT, 1:nT)*R2(1:nT, tt) - Tar(1:nT, tt);
            errR = ztR(tt) - fOut(tt);
            errNR = ztNR(tt) - fOut(tt);
%             chi1(nRun) = chi1(nRun) + mean(err1.^2);
%             chi2(nRun) = chi2(nRun) + mean(err2.^2);
            k = PJ*R2(1:nT, tt);
            kW = PW*R2(:, tt);
            kWNR = PWNR*R2(nT+1: N, tt);
            kWR = PWR*R2(1: nT, tt);
            rPr = R2(1:nT, tt)'*k;
            rPrW = R2(:, tt)'*kW;
            rPrWNR = R2(nT+1: N, tt)'*kWNR;
            rPrWR = R2(1:nT, tt)'*kWR;
            c = 1.0/(1.0 + rPr);
            cW = 1.0/(1.0 + rPrW);
            cWR = 1.0/(1.0 + rPrWR);
            cWNR = 1.0/(1.0 + rPrWNR);
            PJ = PJ - c*(k*k');
            PW = PW - cW*(kW*kW');
            PWR = PWR - cWR*(kWR*kWR');
            PWNR = PWNR - cWNR*(kWNR*kWNR');
            w = w - cW*err1*kW;
            wR = wR - cWR*errR*kWR;
            wNR = wNR - cWNR*errNR*kWNR;
            J2(1:nT, 1:nT) = J2(1:nT, 1:nT) - c*err2*k';
        end
    end
    
    %     figure(1)
    %     for i = 1:10
    %         subplot(5, 2, i)
    %         plot(t, J1(i, :)*R1(:, tt)+input5(i, tt), 'r');
    %         hold on;
    %         plot(t, J2(i, :)*R2(:, tt), 'b');
    %         hold off
    %     end
    %     pause(0.001);
    
    figure(2) 
    subplot(4, 1, 1)
%     hold on;
    if (fIn1 == log(4))
        plot(t, exp(fIn1), 'g');
    else
        plot(t, exp(fIn1), 'c');
    end
    xlim([0 T]);
    legend('auditory input pulse, log(presented amplitude)');
    subplot(4, 1, 2)
    plot(t, zt, 'b');
    hold on;
    plot(t, fOut, 'r');
    hold off;
    legend('overall network output','ideal output')
    xlim([0 T]); ylim([-0.5 1.5]);
    subplot(4, 1, 3)
    plot(t, ztR, 'b');
    hold on;
    plot(t, fOut, 'r');
    hold off;
    xlim([0 T]); ylim([-0.5 1.5]);
    legend('Responsive cells decoded','ideal output')
    subplot(4, 1, 4)
    plot(t, ztNR, 'b');
    hold on;
    plot(t, fOut, 'r');
    hold off;
    xlim([0 T]); ylim([-0.5 1.5]);
    legend('Nom Non Resp cells decoded','ideal output')
    pause(0.001);
    
    figure(3)
    for i = 1:10
        subplot(10, 2, 2*(i - 1)+1)
        plot(t, R2(i, :), 'g');
        xlim([0 T]);ylim([-1 1]);
        title('Repsonsive')
        subplot(10, 2, 2*i)
        plot(t, R2(nT+i, :)./2, 'r');
        xlim([0 T]);ylim([-1 1]);
        title('NNR')
    end
    pause(0.001);
    
    ampVec(nRun) = ampIn;
    OutInt(nRun) = sum(zt(1000+dur:1500+dur));
    OutIntR(nRun) = sum(ztR(1000+dur:1500+dur));
    OutIntNR(nRun) = sum(ztNR(1000+dur:1500+dur));
end

% TP, FP, TN, FN: performance metrics
positives = (OutInt>200);
negatives = (OutInt<=200);

actual_positives = (ampVec==log(4));
actual_negatives = (ampVec~=log(4));

true_positives = sum(positives & actual_positives)
false_positives = sum(positives -(positives & actual_positives))

true_negatives = sum(negatives & actual_negatives)
false_negatives = sum(negatives -(negatives & actual_negatives))

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * precision * recall / (precision + recall)
pCorrTot = 100*(1-(sum((positives-actual_positives) .^ 2)/length(positives)))
sensitivity = true_positives/sum(actual_positives)

positivesNR = (OutIntNR>200);
negativesNR = (OutIntNR<=200);

actual_positivesNR = (ampVec==log(4));
actual_negativesNR = (ampVec~=log(4));

true_positivesNR = sum(positivesNR & actual_positivesNR);
false_positivesNR = sum(positivesNR -(positivesNR & actual_positivesNR));

true_negativesNR = sum(negativesNR & actual_negativesNR);
false_negativesNR = sum(negativesNR -(negativesNR & actual_negativesNR));


precisionNR = true_positivesNR / (true_positivesNR + false_positivesNR);
recallNR = true_positivesNR / (true_positivesNR + false_negativesNR);
f1NR = 2 * precisionNR * recallNR / (precisionNR + recallNR);
pCorrNR = 100*(1-(sum((positivesNR-actual_positivesNR) .^ 2)/length(positivesNR)))
sensitivityNR = true_positivesNR/sum(actual_positivesNR);

figure(4)
[hR hxR] = hist(mean(R2(1:nT, :), 2), 20);
[hNR hxNR] = hist(mean(R2(nT+1:N, :), 2), 20);
hR = hR./sum(hR);
hNR = hNR./sum(hNR);
semilogy(hxR, hR, 'gs-');
hold on;
semilogy(hxNR, hR, 'rs-');
hold off;
legend('R cells','NR cells');
axis square; grid on;
xlim([-0.62 0.62]);
ylim([0.005 0.2]);

% % chi2 = chi2/floor(length(t)/dt);
varData = var(reshape(R2, N*length(t), 1));
chi2 = chi2/(sqrt(N*length(t))*varData);
figure(5)
plot(chi1, '.-');
hold on;
plot(chi2, '.-');
hold off;
xlabel('learning steps');
ylabel('chi^2 error during learning');
axis square;

figure(5)
[hR J1R] = hist(mean(J1, 2), 20);
[hNR J2NR] = hist(mean(J2, 2), 20);
hR = hR./sum(hR);
hNR = hNR./sum(hNR);
semilogy(J1R, hR, 'gs-');
hold on;
semilogy(J2NR, hR, 'rs-');
hold off;
legend('Teacher','Doer');
axis square; grid on;
xlim([-0.004 0.004]);
ylim([0.0005 0.25]);

% 15b: each frequency: 100*(true_positives + false_positives)/(number of trials)

% time_steps = total_time/dt
% R2 = rates of units in doer
% free_start = start of a non-task/non-behav period as a baseline
% free_end = end of a non-task/non-behav period as a baseline
% stim_start = start of period where stimuli is present
% stim_end = end of period where stimuli is present

% pre_behav_start = start of period before response is expected (pick what they used in paper)
% pre_behav_end = end of period pre-behavior

% free_period = zeros(length(t),1);
% free_period(1:1000) = 1 / (1000 - 1 + 1);
% response_during_free = R2 * free_period;
% 
% stim_period = zeros(length(t),1);
% stim_period(1000:2000) = 1 / (2000 - 1000 + 1);
% response_during_stim = R2 * stim_period;
% 
% change_during_stim = response_during_stim - response_during_free;
% figure; hist(change_during_stim);
% %histogram. One hopes bimodal.
% 
% % in paper they did this ramping thing... we could do that, but for a first pass:
% pre_behav_period = zeros(length(t),1);
% pre_behav_period(2000:2500) = 1 / (2500 - 2000 + 1);
% response_pre_behav = R2 * pre_behav_period;
% change_during_stim = response_pre_behav - response_during_free;
% figure; hist(change_during_stim);