%% Estimation and Learning in Aerospace (2023/2024)
% Model identification and optimization of a multirotor UAV dynamics

clearvars;
close all;
clc;
addpath('datasets','common','common/simulator-toolbox',...
    'common/simulator-toolbox/attitude_library',...
    'common/simulator-toolbox/trajectory_library');

%% TASK 1 Model estimation

% Initial model:
% state: longitudinal velocity, pitch rate, pitch angle; 
% input: normalised pitching moment; 
% outputs: state and longitudinal acceleration

Xu = -0.1068;
Xq = 0.1192;
Mu = -5.9755;
Mq = -2.6478;
Xd = -10.1647;
Md = 450.71;

A = [Xu, Xq, -9.81; Mu, Mq, 0; 0, 1, 0];
B = [Xd; Md; 0];
C = [1, 0, 0; 0, 1, 0; 0, 0, 1; Xu, Xq, 0]; 
D = [0; 0; 0; Xd];

% Noise is considered
noise.Enabler = 1;
noise.pos_stand_dev = noise.Enabler * 0.0011;  % [m]
noise.vel_stand_dev = noise.Enabler * 0.01;  % [m/s]
noise.attitude_stand_dev = noise.Enabler * deg2rad(0.33);  % [rad]
noise.ang_rate_stand_dev = noise.Enabler * deg2rad(1);  % [rad/s]

% Delays
delay.position_filter = 1;
delay.attitude_filter = 1;
delay.mixer = 1;

% Load controller parameters
parameters_controller  

% M injection example 
% sweep: first column time vector
% second column time history of pitching moment
load ExcitationM
SetPoint = [0,0];

% time and simulation time
t = ExcitationM(:,1);
simulation_time = t(end)-t(1);

% Launch SIMULATOR
simout = sim('Simulator_Single_Axis.slx'); 
sim_time = 0:sample_time:simulation_time;  % Time vector from Simulink output

% Prepare the data for identification
outputs = [simout.q simout.ax];
inputs = simout.Mtot;
data = iddata(outputs, inputs, sample_time, 'Name', 'QuadRotor');
data.InputName = 'Total Pitching Moment';
data.InputUnit = 'N/m';
data.OutputName = {'Pitch Rate', 'Longitudinal Acceleration'};
data.OutputUnit = {'rad/s', 'm/s^2'};
data.Tstart = t(1);
data.TimeUnit = 's';

% Delete temporary files
if exist('slprj','dir')
    rmdir('slprj', 's')                                                    
end

% Frequency domain identification
data_f = fft(data);
parameters_guess = [0,0,0,0,0,0];
init_sys = idgrey(@Dynamics,parameters_guess,'c');

% Perform the parameter estimation
options = greyestOptions('Display', 'on'); 
estimated_sys = greyest(data_f, init_sys, options);

Xu_est = estimated_sys.A(1,1);
Xq_est = estimated_sys.A(1,2);
Mu_est = estimated_sys.A(2,1);
Mq_est = estimated_sys.A(2,2);
Xd_est = estimated_sys.B(1);
Md_est = estimated_sys.B(2);

paramCovariance = getcov(estimated_sys);
paramVariance = diag(paramCovariance);
totalVariance = sum(paramVariance);

Xu_std = paramVariance(1);
Xq_std = paramVariance(2);
Mu_std = paramVariance(3);
Mq_std = paramVariance(4);
Xd_std = paramVariance(5);
Md_std = paramVariance(6);

% Load estimated model
A = [Xu_est, Xq_est, -9.81; Mu_est, Mq_est, 0; 0, 1, 0];
B = [Xd_est; Md_est; 0];
C = [1, 0, 0; 0, 1, 0; 0, 0, 1; Xu_est, Xq_est, 0]; 
D = [0; 0; 0; Xd_est];

% Validate the estimated model
figure()
compare(data_f, estimated_sys);

%% TASK 2.1 Input optimization

disp('Starting the Optimization')

costfunction = 1; % choose 1, 2 or 3 (see slides) 
ms = MultiStart();

% Options
opts = optimoptions('fmincon');
opts.Algorithm = 'interior-point';
opts.MaxIterations = 500;
opts.MaxFunctionEvaluations = 500;
opts.ConstraintTolerance = 1e-9;
opts.StepTolerance = 1e-9;
opts.OptimalityTolerance = 1e-9;
opts.Display = 'iter';

% Initial conditions and boundaries
theta0 = [Xu_est,Xq_est,Mu_est,Mq_est,Xd_est,Md_est];
eta0 = [25,50,90*rand];
lb = [0,0,0];
fmax = pi/sample_time;
ub = [fmax,fmax,90];

% Problem definition
objective_function = @(eta) optimization_eta(eta,sample_time,theta0,costfunction);
problem = createOptimProblem("fmincon",'x0',eta0,'objective',objective_function,...
    'lb',lb,'ub',ub','nonlcon',@const,'options',opts);

% Solution
tic;
[x_opt, fval, exitflag, output, solutions] = run(ms, problem, 30); % Run
time_of_run = toc;

% optimized model
[J,theta,optimized_sys,data_f,paramVariance]=run_simulation(x_opt,sample_time,...
    theta0,costfunction);

figure()
compare(data_f, optimized_sys);

%% TASK 2.2 MonteCarlo simulation

disp('Starting the MonteCarlo Simulation');

n=30; 
rng(0);  % Set the random seed for reproducibility

% Generate the random field, transform the random numbers to the desired distribution 
mu = theta0';
sigma = paramVariance;
random_field = mu + sigma .* randn(length(mu), n);

% Initialization of empty verctors and matrices
eta_mc = zeros(3,n);
fval_mc = zeros(1,n);
Theta = zeros(n,6);
avg = zeros(1,10);
wc = zeros(1,10);

% Running the MonteCarlo Simulation
for i = 1:n
    theta0 = random_field(:,i);
    objective_function = @(eta) optimization_eta(eta,sample_time,theta0,costfunction);
    [eta_mc(:,i),fval_mc(i),exitflag,output,solutions] = fmincon(objective_function,...
        x_opt,[],[],[],[],lb,ub,@const,opts);
    [~,Theta(i,:),~,~,~] = run_simulation(eta_mc(:,i),sample_time,theta0,costfunction);
end

for i = 1 : n
    for j = 1 : n
        % Generating the j-th model 
        Xu_est = Theta(j,1);
        Xq_est = Theta(j,2);
        Mu_est = Theta(j,3);
        Mq_est = Theta(j,4);
        Xd_est = Theta(j,5);
        Md_est = Theta(j,6);
        A = [Xu_est, Xq_est, -9.81; Mu_est, Mq_est, 0; 0, 1, 0];
        B = [Xd_est; Md_est; 0];
        C = [1, 0, 0; 0, 1, 0; 0, 0, 1; Xu_est, Xq_est, 0]; 
        D = [0; 0; 0; Xd_est];
        % Running the estimation to retrieve J
        [J(j),~,~,~,~]=run_simulation(eta_mc(:,i),sample_time,theta0,costfunction);
    end
    avg(i) = mean(J); % Average for the i-th input
    wc(i) = max(J); % Worst Case for the i-th input
end

% Best worst-case performance
[best_wc, ind_wc] = min(wc);
eta_wc = eta_mc(:,ind_wc);

[J_wc,theta_wc,estimated_sys_wc,data_f,parVariance_wc] = run_simulation(eta_wc,...
    sample_time,theta0,costfunction);

% Validate the estimated model
figure()
compare(data_f, estimated_sys_wc);

% Best average performance
[best_av, ind_av] = min(avg);
eta_avg = eta_mc(:,ind_av);

[J_avg,theta_avg,estimated_sys_avg,data_f,parVariance_avg] = run_simulation(eta_avg,...
    sample_time,theta0,costfunction);

% Validate the estimated model
figure()
compare(data_f, estimated_sys_avg);

save('data_work_J2')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Functions


function [A, B, C, D] = Dynamics(parameters,varargin)

Xu = parameters(1);
Xq = parameters(2);
Mu = parameters(3);
Mq = parameters(4);
Xd = parameters(5);
Md = parameters(6);

A =     [Xu, Xq, -9.81; 
         Mu, Mq, 0; 
         0, 1, 0];
    
B = [Xd; 
     Md; 
     0];

C = [0, 1, 0;    
     Xu, Xq, 0]; 

D = [0;          
     Xd];        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function J = optimization_eta(eta,sample_time,theta0,costfunction)

[J,~,~,~,~] = run_simulation(eta,sample_time,theta0,costfunction);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c,ceq] = const(y)

f1 = y(1);
f2 = y(2);
c = f1 - f2;
ceq = 0;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [J,theta,estimated_sys,data_f,paramVariance] = run_simulation(eta,...
    sample_time,theta0,costfunction)

f1=eta(1);f2=eta(2);T=eta(3);
t=0:sample_time:T;
u=0.1*sin(2*pi.*(f1+(f2-f1).*t/T).*t);
ExcitationM=[t;u]';
assignin('base', 'ExcitationM', ExcitationM);
simulation_time=T;
sim_time = 0:sample_time:simulation_time; % Time vector from Simulink output
simout= sim('Simulator_Single_Axis_Amine.slx'); 
outputs = [simout.q simout.ax];
inputs=simout.Mtot;

data = iddata(outputs, inputs, sample_time, 'Name', 'QuadRotor');
data.InputName = 'Total Pitching Moment';
data.InputUnit = 'N/m';
data.OutputName = {'Pitch Rate', 'Longitudinal Acceleration'};
data.OutputUnit = {'rad/s', 'm/s^2'};
data.Tstart = t(1);
data.TimeUnit = 's';

% Delete temporary files
if exist('slprj','dir')
    rmdir('slprj', 's')                                                    
end

% Frequency domain identification
data_f = fft(data);
parameters_guess = theta0;
init_sys = idgrey(@Dynamics,parameters_guess,'c');

% Perform the parameter estimation
options = greyestOptions('Display', 'off'); 
estimated_sys = greyest(data_f, init_sys, options);

paramCovariance = getcov(estimated_sys);
paramVariance = diag(paramCovariance);

Xu_est = estimated_sys.A(1,1);
Xq_est = estimated_sys.A(1,2);
Mu_est = estimated_sys.A(2,1);
Mq_est = estimated_sys.A(2,2);
Xd_est = estimated_sys.B(1);
Md_est = estimated_sys.B(2);

theta=[Xu_est,Xq_est,Mu_est,Mq_est,Xd_est,Md_est];

% Cost Function 1
if costfunction == 1
    J = sum(paramVariance);

% Cost Function 2
elseif costfunction == 2
    J=abs(paramVariance(1)/Xu_est)+abs(paramVariance(2)/Xq_est)+...
        abs(paramVariance(3)/Mu_est)+abs(paramVariance(4)/Mq_est)+...
        abs(paramVariance(5)/Xd_est)+abs(paramVariance(6)/Md_est);

% Cost Function 3
elseif costfunction == 3
    weights = [1/22;4/22;1/22;2/22;3/22;6/22];

    J = abs(paramVariance(1)/Xu_est) * weights(1) + ...
    abs(paramVariance(2)/Xq_est) * weights(2) + ...
    abs(paramVariance(3)/Mu_est) * weights(3) + ...
    abs(paramVariance(4)/Mq_est) * weights(4) + ...
    abs(paramVariance(5)/Xd_est) * weights(5) + ...
    abs(paramVariance(6)/Md_est) * weights(6);
end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
