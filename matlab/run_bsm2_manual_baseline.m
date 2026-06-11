%% run_bsm2_manual_baseline.m - Manual BSM2 baseline, no Python handshake
%
% Purpose
%   Run the default closed-loop BSM2 plant once from day 0 to day 609 and
%   export one reusable baseline for all four SAC agents.
%
% Why this is faster than RL_main_simple.m
%   The RL runner pauses every control interval and waits for Python. This
%   baseline script does not pause for decisions; it lets Simulink run
%   continuously with the manual/default BSM2 controls.
%
% Outputs
%   prog_RL/results/bsm2_manual_baseline_timeseries.csv
%   prog_RL/results/bsm2_manual_baseline_summary.csv
%   prog_RL/results/bsm2_manual_baseline_official_summary.csv
%
% Usage in MATLAB
%   init_bsm2
%   run('C:\Users\marin\Documents\Universidade\Thesis work\prog_RL\matlab\run_bsm2_manual_baseline.m')

%% Project paths
RL_DIR   = fileparts(fileparts(mfilename('fullpath')));
THESIS_DIR = fileparts(RL_DIR);
UNI_DIR    = fileparts(THESIS_DIR);
BSM2_DIR   = fullfile(THESIS_DIR, 'BSM2_R2019b');
if ~isfolder(BSM2_DIR)
    BSM2_DIR = fullfile(UNI_DIR, 'BSM2_R2019b');
end
if ~isfolder(BSM2_DIR)
    error('[run_bsm2_manual_baseline] BSM2_R2019b folder not found. Checked Thesis work and Universidade folders.');
end
OUT_DIR  = fullfile(RL_DIR, 'results');

if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

baselinePath = fullfile(OUT_DIR, 'bsm2_manual_baseline_timeseries.csv');

postprocessOnly = evalin('base', ...
    'exist(''BSM2_BASELINE_POSTPROCESS_ONLY'', ''var'') == 1 && BSM2_BASELINE_POSTPROCESS_ONLY');
if postprocessOnly
    if exist(baselinePath, 'file') ~= 2
        error('[baseline] Post-process requested, but baseline CSV does not exist: %s', baselinePath);
    end
    fprintf('[baseline] Post-processing existing baseline CSV: %s\n', baselinePath);
    baseline = readtable(baselinePath);
    if evalin('base', 'exist(''BSM2_BASELINE_ELAPSED_SECONDS'', ''var'') == 1')
        elapsedSeconds = evalin('base', 'BSM2_BASELINE_ELAPSED_SECONDS');
    elseif exist('elapsedSeconds', 'var') == 1
        % Keeps the value if this mode is run immediately after a summary error.
        elapsedSeconds = elapsedSeconds;
    else
        elapsedSeconds = NaN;
    end
    write_baseline_summaries(baseline, OUT_DIR, elapsedSeconds);
    return
end

cd(BSM2_DIR);
addpath(BSM2_DIR);
addpath(fullfile(RL_DIR, 'matlab'));

model = 'bsm2_cl';

%% Stop and reload model cleanly
if bdIsLoaded(model)
    try
        simStatus = get_param(model, 'SimulationStatus');
    catch
        simStatus = 'stopped';
    end

    if ~strcmpi(simStatus, 'stopped')
        fprintf('[baseline] Stopping existing %s simulation...\n', model);
        set_param(model, 'SimulationCommand', 'stop');
        tStop = tic;
        while ~strcmpi(get_param(model, 'SimulationStatus'), 'stopped')
            pause(0.2);
            if toc(tStop) > 60
                error('[baseline] Timeout waiting for %s to stop.', model);
            end
        end
    end

    close_system(model, 0);
end

load_system(model);

%% Initialize BSM2 defaults if available
if exist('init_bsm2', 'file') == 2
    fprintf('[baseline] Running init_bsm2...\n');
    init_bsm2;
else
    warning('[baseline] init_bsm2 not found on path. Continuing with current workspace values.');
end

%% Manual/default control values used by the thesis agents
assignin('base', 'carb1', 2.0);      % Qec default, m3/d
assignin('base', 'Qintr', 61944.0);  % Qint default, m3/d
assignin('base', 'SO4ref', 2.0);     % DO setpoint default, mg/L
assignin('base', 'Qw', 300.0);       % Qw low default, m3/d
assignin('base', 'Qw_low', 300.0);
assignin('base', 'Qw_high', 450.0);

%% Fastest practical simulation setup
dt = 15 / (24 * 60);

set_param(model, 'StartTime', '0');
set_param(model, 'StopTime', '609');
set_param(model, 'OutputTimes', '[0:(1/96):609]');
try
    set_param(model, 'PauseTimes', '[]');
catch pauseErr
    fprintf('[baseline] PauseTimes not supported by this model (%s). Continuing without it.\n', pauseErr.message);
end

% Rapid accelerator is intentionally skipped for BSM2 closed loop. This
% model contains algebraic loops, so standalone rapid-accelerator code
% generation fails. Normal accelerator is the fastest stable mode here.
set_param(model, 'SimulationMode', 'accelerator');
simMode = 'accelerator';

fprintf('[baseline] Starting manual BSM2 run: t=0..609 days, mode=%s\n', simMode);
tRun = tic;
try
    run_simulation_with_progress(model, 609);
catch simErr
    rethrow(simErr);
end
elapsedSeconds = toc(tRun);
fprintf('[baseline] Simulation finished in %.1f seconds.\n', elapsedSeconds);

%% Collect BSM2 workspace outputs
requiredVars = {'t', 'reac1', 'reac2', 'reac3', 'reac4', 'reac5', 'settler'};
for i = 1:numel(requiredVars)
    if evalin('base', sprintf('exist(''%s'', ''var'')', requiredVars{i})) ~= 1
        error('[baseline] Required workspace variable missing after sim: %s', requiredVars{i});
    end
end

t       = evalin('base', 't');
reac1   = evalin('base', 'reac1');
reac2   = evalin('base', 'reac2');
reac3   = evalin('base', 'reac3');
reac4   = evalin('base', 'reac4');
reac5   = evalin('base', 'reac5');
settler = evalin('base', 'settler');

if evalin('base', 'exist(''DYNINFLUENT_BSM2'', ''var'')') == 1
    influent = evalin('base', 'DYNINFLUENT_BSM2');
else
    influent = [];
end

N = min([numel(t), size(reac1, 1), size(reac2, 1), size(reac3, 1), ...
         size(reac4, 1), size(reac5, 1), size(settler, 1)]);
t = t(1:N);
reac1 = reac1(1:N, :);
reac2 = reac2(1:N, :);
reac3 = reac3(1:N, :);
reac4 = reac4(1:N, :);
reac5 = reac5(1:N, :);
settler = settler(1:N, :);
if ~isempty(influent)
    influent = influent(1:min(N, size(influent, 1)), :);
end

%% Build reusable baseline table
SNO_1 = reac1(:, 9);
SNO_2 = reac2(:, 9);
SNO_3 = reac3(:, 9);
SNO_5 = reac5(:, 9);
SNH_2 = reac2(:, 10);
SNH_4 = reac4(:, 10);
SNH_5 = reac5(:, 10);
SO_4  = reac4(:, 8);
SO_5  = reac5(:, 8);
SND_5 = reac5(:, 11);
TSS_3 = reac3(:, 14);
TSS_5 = reac5(:, 14);

Qec_default = 2.0;
Qint_default = 61944.0;
SO4ref_default = 2.0;
Qw_default = settler(:, 22);

if ~isempty(influent) && size(influent, 2) >= 16
    Flow = influent(:, 16);
    if numel(Flow) < N
        Flow(end+1:N, 1) = Flow(end);
    end
    SS_in = influent(:, 3);
    SI_in = influent(:, 2);
    SNH_in = influent(:, 11);
    if numel(SS_in) < N
        SS_in(end+1:N, 1) = SS_in(end);
        SI_in(end+1:N, 1) = SI_in(end);
        SNH_in(end+1:N, 1) = SNH_in(end);
    end
else
    Flow = repmat(20648.0, N, 1);
    SS_in = nan(N, 1);
    SI_in = nan(N, 1);
    SNH_in = nan(N, 1);
end

CODTN = (SS_in + SI_in) ./ max(SNH_in, 1.0);
CODTN = min(max(CODTN, 0), 100);

% Thesis reward proxy, matching prog_RL/core/reward.py.
J_MANUAL = 1293523.0;
AE = repmat(4000.0, N, 1);
PE = repmat(0.05 * Qint_default, N, 1);
EC = repmat(Qec_default * 400000.0 / 1000000.0, N, 1);
EQI = Flow .* (30.0 .* SNH_2 + 10.0 .* SNO_2) ./ 1000.0;
J = 200.0 .* EQI + 40.0 .* AE + 3.0 .* PE + EC;
ratio = J ./ J_MANUAL;
reward = max(-5.0 .* ratio + 5.0, -1.0);

baseline = table(t(:), SNO_1, SNO_2, SNO_3, SNO_5, SNH_2, SNH_4, SNH_5, ...
    SO_4, SO_5, SND_5, TSS_3, TSS_5, Flow(:), CODTN(:), ...
    repmat(Qec_default, N, 1), repmat(Qint_default, N, 1), ...
    repmat(SO4ref_default, N, 1), Qw_default, EQI, AE, PE, EC, J, ratio, reward, ...
    'VariableNames', {'time','SNO_1','SNO_2','SNO_3','SNO_5','SNH_2','SNH_4','SNH_5', ...
    'SO_4','SO_5','SND_5','TSS_3','TSS_5','Flow','CODTN', ...
    'Qec','Qint','SO4ref','Qw','EQI','AE','PE','EC','J','ratio','reward'});

writetable(baseline, baselinePath);
write_baseline_summaries(baseline, OUT_DIR, elapsedSeconds);

function write_baseline_summaries(baseline, OUT_DIR, elapsedSeconds)
    %% Summary for the standard BSM2 evaluation period
    baselinePath = fullfile(OUT_DIR, 'bsm2_manual_baseline_timeseries.csv');
    evalMask = baseline.time >= 245 & baseline.time <= 609;
    evalData = baseline(evalMask, :);

    metricNames = {'rows'; 'start_day'; 'stop_day'; 'elapsed_seconds'; ...
        'mean_EQI'; 'mean_AE'; 'mean_PE'; 'mean_EC'; 'mean_J'; 'mean_ratio'; 'mean_reward'; ...
        'mean_SNO_2'; 'mean_SNO_3'; 'mean_SNO_5'; 'mean_SNH_2'; 'mean_SNH_5'; ...
        'mean_SO_4'; 'mean_SO_5'; 'mean_TSS_5'; 'mean_Qec'; 'mean_Qint'; 'mean_Qw'};
    metricValues = [size(evalData, 1); 245; 609; elapsedSeconds; ...
        mean(evalData.EQI, 'omitnan'); mean(evalData.AE, 'omitnan'); ...
        mean(evalData.PE, 'omitnan'); mean(evalData.EC, 'omitnan'); ...
        mean(evalData.J, 'omitnan'); mean(evalData.ratio, 'omitnan'); ...
        mean(evalData.reward, 'omitnan'); mean(evalData.SNO_2, 'omitnan'); ...
        mean(evalData.SNO_3, 'omitnan'); mean(evalData.SNO_5, 'omitnan'); ...
        mean(evalData.SNH_2, 'omitnan'); mean(evalData.SNH_5, 'omitnan'); ...
        mean(evalData.SO_4, 'omitnan'); mean(evalData.SO_5, 'omitnan'); ...
        mean(evalData.TSS_5, 'omitnan'); mean(evalData.Qec, 'omitnan'); ...
        mean(evalData.Qint, 'omitnan'); mean(evalData.Qw, 'omitnan')];

    summary = table(metricNames, metricValues, 'VariableNames', {'metric', 'value'});
    summaryPath = fullfile(OUT_DIR, 'bsm2_manual_baseline_summary.csv');
    writetable(summary, summaryPath);

    %% Official-style BSM2 closed-loop metrics
    % These follow the plant-performance definitions in the official BSM2
    % closed-loop package, especially perf_plant_bsm2.m:
    %   - standard evaluation window: days 245..609
    %   - EQI with BSM2/BSM1 pollutant weights
    %   - OCI with aeration, pumping, mixing, sludge, carbon, methane/heating
    %   - effluent violation time/counts and 95th percentiles
    try
        officialSummary = compute_official_bsm2_summary_from_base(elapsedSeconds);
    catch officialErr
        officialSummary = table( ...
            {'official_metrics_available'; 'elapsed_seconds'}, ...
            [0; elapsedSeconds], ...
            {'Official BSM2 summary failed after simulation'; officialErr.message}, ...
            'VariableNames', {'metric','value','note'});
    end
    officialSummaryPath = fullfile(OUT_DIR, 'bsm2_manual_baseline_official_summary.csv');
    writetable(officialSummary, officialSummaryPath);

    fprintf('[baseline] Wrote time-series: %s\n', baselinePath);
    fprintf('[baseline] Wrote summary    : %s\n', summaryPath);
    fprintf('[baseline] Wrote official   : %s\n', officialSummaryPath);
    fprintf('[baseline] Evaluation rows days 245..609: %d\n', size(evalData, 1));
end

function summary = compute_official_bsm2_summary_from_base(elapsedSeconds)
    required = {'t','in','reac1','reac2','reac3','reac4','reac5','settler', ...
        'effluent','sludge','rec','primaryout','thickenerout','digesterin', ...
        'digesterout','dewateringout','storageout','qpassplant','qpassAS'};
    missing = {};
    for k = 1:numel(required)
        if evalin('base', sprintf('exist(''%s'', ''var'')', required{k})) ~= 1
            missing{end+1} = required{k}; %#ok<AGROW>
        end
    end

    if ~isempty(missing)
        summary = table( ...
            {'official_metrics_available'; 'elapsed_seconds'}, ...
            [0; elapsedSeconds], ...
            {'Missing required BSM2 workspace variables after simulation'; strjoin(missing, ', ')}, ...
            'VariableNames', {'metric','value','note'});
        return
    end

    t = evalin('base', 't');
    starttime = 245;
    stoptime = 609;
    startindex = max(find(t <= starttime)); %#ok<MXFND>
    stopindex = min(find(t >= stoptime)); %#ok<MXFND>
    time_eval = t(startindex:stopindex);
    timevector = time_eval(2:end) - time_eval(1:(end-1));
    totalt = time_eval(end) - time_eval(1);

    inpart = evalin('base', 'in');
    reac1part = evalin('base', 'reac1');
    reac2part = evalin('base', 'reac2');
    reac3part = evalin('base', 'reac3');
    reac4part = evalin('base', 'reac4');
    reac5part = evalin('base', 'reac5');
    settlerpart = evalin('base', 'settler');
    effluentpart = evalin('base', 'effluent');
    sludgepart = evalin('base', 'sludge');
    recpart = evalin('base', 'rec');
    primarypart = evalin('base', 'primaryout');
    thickenerpart = evalin('base', 'thickenerout');
    digesterinpart = evalin('base', 'digesterin');
    digesteroutpart = evalin('base', 'digesterout');
    dewateringoutpart = evalin('base', 'dewateringout');
    storagepart = evalin('base', 'storageout');
    qpassplantpart = evalin('base', 'qpassplant');
    qpassASpart = evalin('base', 'qpassAS');

    inpart = inpart(startindex:(stopindex-1), :);
    reac1part = reac1part(startindex:(stopindex-1), :);
    reac2part = reac2part(startindex:(stopindex-1), :);
    reac3part = reac3part(startindex:(stopindex-1), :);
    reac4part = reac4part(startindex:(stopindex-1), :);
    reac5part = reac5part(startindex:(stopindex-1), :);
    settlerpart = settlerpart(startindex:(stopindex-1), :);
    effluentpart = effluentpart(startindex:(stopindex-1), :);
    sludgepart = sludgepart(startindex:(stopindex-1), :);
    recpart = recpart(startindex:(stopindex-1), :);
    primarypart = primarypart(startindex:(stopindex-1), :);
    thickenerpart = thickenerpart(startindex:(stopindex-1), :);
    digesterinpart = digesterinpart(startindex:(stopindex-1), :);
    digesteroutpart = digesteroutpart(startindex:(stopindex-1), :);
    dewateringoutpart = dewateringoutpart(startindex:(stopindex-1), :);
    storagepart = storagepart(startindex:(stopindex-1), :);
    qpassplantpart = qpassplantpart(startindex:(stopindex-1), :);
    qpassASpart = qpassASpart(startindex:(stopindex-1), :);

    i_XB = bsm2_base_value('i_XB', 0.08);
    i_XP = bsm2_base_value('i_XP', 0.06);
    f_P = bsm2_base_value('f_P', 0.08);
    ACTIVATE = bsm2_base_value('ACTIVATE', 0);
    if ACTIVATE > 0.5
        warning('[baseline] Dummy-variable ACTIVATE mode is not included in the compact official summary.');
    end

    BSS = 2;
    BCOD = 1;
    BNKj = 30;
    BNO = 10;
    BBOD5 = 2;

    totalCODemax = 100;
    totalNemax = 18;
    SNHemax = 4;
    TSSemax = 30;
    BOD5emax = 10;

    Qevec = effluentpart(:,15) .* timevector;
    Qetot = sum(Qevec);
    Qeav = Qetot / totalt;

    SSevec = effluentpart(:,2) .* Qevec;
    XIevec = effluentpart(:,3) .* Qevec;
    XSevec = effluentpart(:,4) .* Qevec;
    XBHevec = effluentpart(:,5) .* Qevec;
    XBAevec = effluentpart(:,6) .* Qevec;
    XPevec = effluentpart(:,7) .* Qevec;
    SNOevec = effluentpart(:,9) .* Qevec;
    SNHevec = effluentpart(:,10) .* Qevec;
    SNDevec = effluentpart(:,11) .* Qevec;
    XNDevec = effluentpart(:,12) .* Qevec;
    TSSevec = effluentpart(:,14) .* Qevec;

    qpassplantvec = qpassplantpart(:,15) .* timevector;
    qpassASvec = qpassASpart(:,15) .* timevector;
    SSbypassplantvec = qpassplantpart(:,2) .* qpassplantvec;
    XSbypassplantvec = qpassplantpart(:,4) .* qpassplantvec;
    XBHbypassplantvec = qpassplantpart(:,5) .* qpassplantvec;
    XBAbypassplantvec = qpassplantpart(:,6) .* qpassplantvec;
    SSbypassASvec = qpassASpart(:,2) .* qpassASvec;
    XSbypassASvec = qpassASpart(:,4) .* qpassASvec;
    XBHbypassASvec = qpassASpart(:,5) .* qpassASvec;
    XBAbypassASvec = qpassASpart(:,6) .* qpassASvec;

    totalNevec2 = (SNOevec + SNHevec + SNDevec + XNDevec + ...
        i_XB .* (XBHevec + XBAevec) + i_XP .* (XPevec + XIevec)) ./ Qevec;
    totalCODevec2 = (effluentpart(:,1).*Qevec + SSevec + XIevec + XSevec + ...
        XBHevec + XBAevec + XPevec) ./ Qevec;
    SNHevec2 = SNHevec ./ Qevec;
    TSSevec2 = TSSevec ./ Qevec;
    BOD5_SSloadvec = 0.25 .* (SSevec - SSbypassplantvec - SSbypassASvec) + ...
        0.65 .* (SSbypassplantvec + SSbypassASvec);
    BOD5_XSloadvec = 0.25 .* (XSevec - XSbypassplantvec - XSbypassASvec) + ...
        0.65 .* (XSbypassplantvec + XSbypassASvec);
    BOD5_XBHloadvec = 0.25 .* (1 - f_P) .* (XBHevec - XBHbypassplantvec - XBHbypassASvec) + ...
        0.65 .* (1 - f_P) .* (XBHbypassplantvec + XBHbypassASvec);
    BOD5_XBAloadvec = 0.25 .* (1 - f_P) .* (XBAevec - XBAbypassplantvec - XBAbypassASvec) + ...
        0.65 .* (1 - f_P) .* (XBAbypassplantvec + XBAbypassASvec);
    BOD5evec2 = (BOD5_SSloadvec + BOD5_XSloadvec + BOD5_XBHloadvec + BOD5_XBAloadvec) ./ Qevec;

    SSe = effluentpart(:,14);
    CODe = effluentpart(:,1) + effluentpart(:,2) + effluentpart(:,3) + ...
        effluentpart(:,4) + effluentpart(:,5) + effluentpart(:,6) + effluentpart(:,7);
    SNKje = effluentpart(:,10) + effluentpart(:,11) + effluentpart(:,12) + ...
        i_XB .* (effluentpart(:,5) + effluentpart(:,6)) + ...
        i_XP .* (effluentpart(:,3) + effluentpart(:,7));
    SNOe = effluentpart(:,9);
    EQIvec = (BSS .* SSe + BCOD .* CODe + BNKj .* SNKje + BNO .* SNOe + BBOD5 .* BOD5evec2) .* Qevec;
    EQI = sum(EQIvec) / (totalt * 1000);

    SSin = inpart(:,14);
    CODin = inpart(:,1) + inpart(:,2) + inpart(:,3) + inpart(:,4) + ...
        inpart(:,5) + inpart(:,6) + inpart(:,7);
    SNKjin = inpart(:,10) + inpart(:,11) + inpart(:,12) + ...
        i_XB .* (inpart(:,5) + inpart(:,6)) + i_XP .* (inpart(:,3) + inpart(:,7));
    SNOin = inpart(:,9);
    BOD5in = 0.65 .* (inpart(:,2) + inpart(:,4) + (1 - f_P) .* (inpart(:,5) + inpart(:,6)));
    Qinvec = inpart(:,15) .* timevector;
    IQIvec = (BSS .* SSin + BCOD .* CODin + BNKj .* SNKjin + BNO .* SNOin + BBOD5 .* BOD5in) .* Qinvec;
    IQI = sum(IQIvec) / (totalt * 1000);

    official_energy = compute_official_oci_components(timevector, totalt, startindex, stopindex, ...
        reac1part, reac2part, reac3part, reac4part, reac5part, settlerpart, sludgepart, ...
        primarypart, thickenerpart, digesterinpart, digesteroutpart, dewateringoutpart, ...
        storagepart, recpart);

    SNH95 = prctile(SNHevec2, 95);
    TN95 = prctile(totalNevec2, 95);
    TSS95 = prctile(TSSevec2, 95);

    [TN_v_time, TN_v_percent, TN_v_count] = violation_stats(totalNevec2 > totalNemax, timevector, totalt);
    [COD_v_time, COD_v_percent, COD_v_count] = violation_stats(totalCODevec2 > totalCODemax, timevector, totalt);
    [SNH_v_time, SNH_v_percent, SNH_v_count] = violation_stats(SNHevec2 > SNHemax, timevector, totalt);
    [TSS_v_time, TSS_v_percent, TSS_v_count] = violation_stats(TSSevec2 > TSSemax, timevector, totalt);
    [BOD_v_time, BOD_v_percent, BOD_v_count] = violation_stats(BOD5evec2 > BOD5emax, timevector, totalt);

    metric = {
        'official_metrics_available';
        'eval_start_day';
        'eval_stop_day';
        'eval_days';
        'elapsed_seconds';
        'IQI_kg_pollution_units_per_d';
        'EQI_kg_pollution_units_per_d';
        'OCI_total';
        'OCI_sludge_cost';
        'OCI_aeration_energy_cost';
        'OCI_pumping_energy_cost';
        'OCI_carbon_cost';
        'OCI_mixing_energy_cost';
        'OCI_heating_cost';
        'OCI_methane_credit';
        'airenergy_kWh_per_d';
        'pumpenergy_kWh_per_d';
        'mixenergy_kWh_per_d';
        'carbon_kgCOD_per_d';
        'methane_kgCH4_per_d';
        'effluent_flow_m3_per_d';
        'SNH95_gN_per_m3';
        'TN95_gN_per_m3';
        'TSS95_gSS_per_m3';
        'TN_violation_days';
        'TN_violation_percent';
        'TN_violation_count';
        'COD_violation_days';
        'COD_violation_percent';
        'COD_violation_count';
        'SNH_violation_days';
        'SNH_violation_percent';
        'SNH_violation_count';
        'TSS_violation_days';
        'TSS_violation_percent';
        'TSS_violation_count';
        'BOD5_violation_days';
        'BOD5_violation_percent';
        'BOD5_violation_count';
    };

    value = [
        1;
        starttime;
        stoptime;
        totalt;
        elapsedSeconds;
        IQI;
        EQI;
        official_energy.OCI;
        official_energy.TSScost;
        official_energy.airenergycost;
        official_energy.pumpenergycost;
        official_energy.carbonmasscost;
        official_energy.mixenergycost;
        official_energy.Heatenergycost;
        official_energy.EnergyfromMethaneperdcost;
        official_energy.airenergyperd;
        official_energy.pumpenergyperd;
        official_energy.mixenergyperd;
        official_energy.carbonmassperd;
        official_energy.Methaneprodperd;
        Qeav;
        SNH95;
        TN95;
        TSS95;
        TN_v_time;
        TN_v_percent;
        TN_v_count;
        COD_v_time;
        COD_v_percent;
        COD_v_count;
        SNH_v_time;
        SNH_v_percent;
        SNH_v_count;
        TSS_v_time;
        TSS_v_percent;
        TSS_v_count;
        BOD_v_time;
        BOD_v_percent;
        BOD_v_count;
    ];

    note = repmat({''}, numel(metric), 1);
    note{1} = 'Official-style metrics based on BSM2 perf_plant_bsm2.m definitions, excluding optional fuzzy risk prompt.';
    summary = table(metric, value, note, 'VariableNames', {'metric','value','note'});
end

function out = compute_official_oci_components(timevector, totalt, startindex, stopindex, ...
        reac1part, reac2part, reac3part, reac4part, reac5part, settlerpart, sludgepart, ...
        primarypart, thickenerpart, digesterinpart, digesteroutpart, dewateringoutpart, ...
        storagepart, recpart)

    VOL1 = bsm2_base_value('VOL1', NaN);
    VOL2 = bsm2_base_value('VOL2', NaN);
    VOL3 = bsm2_base_value('VOL3', NaN);
    VOL4 = bsm2_base_value('VOL4', NaN);
    VOL5 = bsm2_base_value('VOL5', NaN);
    DIM = bsm2_base_value('DIM', [NaN NaN]);
    VOL_P = bsm2_base_value('VOL_P', NaN);
    V_liq = bsm2_base_value('V_liq', NaN);
    CARBONSOURCECONC = bsm2_base_value('CARBONSOURCECONC', 400000);
    SOSAT1 = bsm2_base_value('SOSAT1', 8);
    SOSAT2 = bsm2_base_value('SOSAT2', 8);
    SOSAT3 = bsm2_base_value('SOSAT3', 8);
    SOSAT4 = bsm2_base_value('SOSAT4', 8);
    SOSAT5 = bsm2_base_value('SOSAT5', 8);
    P_atm = bsm2_base_value('P_atm', 1.013);
    R = bsm2_base_value('R', 0.083145);
    T_op = bsm2_base_value('T_op', 308.15);

    TSSreactors_start = (reac1part(1,14)*VOL1 + reac2part(1,14)*VOL2 + reac3part(1,14)*VOL3 + reac4part(1,14)*VOL4 + reac5part(1,14)*VOL5) / 1000;
    TSSreactors_end = (reac1part(end,14)*VOL1 + reac2part(end,14)*VOL2 + reac3part(end,14)*VOL3 + reac4part(end,14)*VOL4 + reac5part(end,14)*VOL5) / 1000;
    TSSsettler_start = sum(settlerpart(1,44:53)) * DIM(1) * DIM(2) / 10 / 1000;
    TSSsettler_end = sum(settlerpart(end,44:53)) * DIM(1) * DIM(2) / 10 / 1000;
    TSSprimary_start = primarypart(1,56) * VOL_P / 1000;
    TSSprimary_end = primarypart(end,56) * VOL_P / 1000;
    TSSdigester_start = digesteroutpart(1,14) * V_liq / 1000;
    TSSdigester_end = digesteroutpart(end,14) * V_liq / 1000;
    TSSstorage_start = storagepart(1,14) * storagepart(1,22) / 1000;
    TSSstorage_end = storagepart(end,14) * storagepart(end,22) / 1000;

    TSSsludgevec = (sludgepart(:,14) / 1000) .* sludgepart(:,15) .* timevector;
    TSSproduced = sum(TSSsludgevec) + TSSreactors_end + TSSsettler_end + TSSprimary_end + TSSdigester_end + TSSstorage_end - ...
        TSSreactors_start - TSSsettler_start - TSSprimary_start - TSSdigester_start - TSSstorage_start;
    TSSproducedperd = TSSproduced / totalt;

    n = evalin('base', 'size(t)');
    kla1in = vectorize_control_signal(bsm2_base_value('kla1in', 0), n);
    kla2in = vectorize_control_signal(bsm2_base_value('kla2in', 0), n);
    kla3in = vectorize_control_signal(bsm2_base_value('kla3in', 0), n);
    kla4in = vectorize_control_signal(bsm2_base_value('kla4in', 0), n);
    kla5in = vectorize_control_signal(bsm2_base_value('kla5in', 0), n);
    carbon1in = vectorize_control_signal(bsm2_base_value('carbon1in', 2), n);
    carbon2in = vectorize_control_signal(bsm2_base_value('carbon2in', 0), n);
    carbon3in = vectorize_control_signal(bsm2_base_value('carbon3in', 0), n);
    carbon4in = vectorize_control_signal(bsm2_base_value('carbon4in', 0), n);
    carbon5in = vectorize_control_signal(bsm2_base_value('carbon5in', 0), n);

    kla1vec = kla1in(startindex:(stopindex-1), :);
    kla2vec = kla2in(startindex:(stopindex-1), :);
    kla3vec = kla3in(startindex:(stopindex-1), :);
    kla4vec = kla4in(startindex:(stopindex-1), :);
    kla5vec = kla5in(startindex:(stopindex-1), :);
    airenergyvec = (SOSAT1*VOL1*kla1vec + SOSAT2*VOL2*kla2vec + SOSAT3*VOL3*kla3vec + SOSAT4*VOL4*kla4vec + SOSAT5*VOL5*kla5vec) / (1.8*1000);
    airenergyperd = sum(airenergyvec .* timevector) / totalt;

    mixenergyunitreac = 0.005;
    mixenergyreac = 24 * ( ...
        length(find(kla1vec < 20))*mixenergyunitreac*VOL1 + ...
        length(find(kla2vec < 20))*mixenergyunitreac*VOL2 + ...
        length(find(kla3vec < 20))*mixenergyunitreac*VOL3 + ...
        length(find(kla4vec < 20))*mixenergyunitreac*VOL4 + ...
        length(find(kla5vec < 20))*mixenergyunitreac*VOL5) * (timevector(1));
    mixenergyAD = 24 * 0.005 * V_liq * totalt;
    mixenergyperd = (mixenergyreac + mixenergyAD) / totalt;

    PF_Qintr = 0.004;
    PF_Qr = 0.008;
    PF_Qw = 0.05;
    PF_Qpu = 0.075;
    PF_Qtu = 0.060;
    PF_Qdo = 0.004;
    pumpenergyvec = PF_Qintr*recpart(:,15) + PF_Qr*settlerpart(:,15) + PF_Qw*settlerpart(:,22) + ...
        PF_Qpu*primarypart(:,36) + PF_Qtu*thickenerpart(:,15) + PF_Qdo*dewateringoutpart(:,36);
    pumpenergyperd = sum(pumpenergyvec .* timevector) / totalt;

    carbon1vec = carbon1in(startindex:(stopindex-1), :);
    carbon2vec = carbon2in(startindex:(stopindex-1), :);
    carbon3vec = carbon3in(startindex:(stopindex-1), :);
    carbon4vec = carbon4in(startindex:(stopindex-1), :);
    carbon5vec = carbon5in(startindex:(stopindex-1), :);
    carbonmassvec = (carbon1vec + carbon2vec + carbon3vec + carbon4vec + carbon5vec) * CARBONSOURCECONC / 1000;
    carbonmassperd = sum(carbonmassvec .* timevector) / totalt;

    Methanevec = digesteroutpart(:,48) ./ digesteroutpart(:,50) * P_atm * 16 / (R*T_op);
    Methaneflowvec = Methanevec .* digesteroutpart(:,51);
    Methaneprodperd = sum(Methaneflowvec .* timevector) / totalt;

    Tdigesterin = (primarypart(:,37).*primarypart(:,36) + thickenerpart(:,16).*thickenerpart(:,15)) ./ ...
        (primarypart(:,36) + thickenerpart(:,15));
    Heatpower = (35 - Tdigesterin) .* digesterinpart(:,27) * 1000 * 4.186 / 86400;
    Heatenergyperd = 24 * sum(Heatpower .* timevector) / totalt;

    out.TSScost = 3 * TSSproducedperd;
    out.airenergycost = airenergyperd;
    out.mixenergycost = mixenergyperd;
    out.pumpenergycost = pumpenergyperd;
    out.carbonmasscost = 3 * carbonmassperd;
    out.EnergyfromMethaneperdcost = 6 * Methaneprodperd;
    out.Heatenergycost = max(0, Heatenergyperd - 7*Methaneprodperd);
    out.OCI = out.TSScost + out.airenergycost + out.mixenergycost + out.pumpenergycost + ...
        out.carbonmasscost - out.EnergyfromMethaneperdcost + out.Heatenergycost;
    out.airenergyperd = airenergyperd;
    out.pumpenergyperd = pumpenergyperd;
    out.mixenergyperd = mixenergyperd;
    out.carbonmassperd = carbonmassperd;
    out.Methaneprodperd = Methaneprodperd;
end

function value = bsm2_base_value(name, defaultValue)
    if evalin('base', sprintf('exist(''%s'', ''var'')', name)) == 1
        value = evalin('base', name);
    else
        value = defaultValue;
    end
end

function out = vectorize_control_signal(invariable, outvecsize)
    if size(invariable, 1) > 1
        out = invariable;
    else
        out = ones(outvecsize) .* invariable;
    end
end

function [days, percent, count] = violation_stats(mask, timevector, totalt)
    idx = find(mask);
    days = min(totalt, numel(idx) * timevector(1));
    percent = min(100, days / totalt * 100);
    if isempty(idx)
        count = 0;
        return
    end
    count = 1 + sum(diff(idx) > 1);
end

function run_simulation_with_progress(model, stopDay)
    set_param(model, 'SimulationCommand', 'start');
    lastPrint = tic;
    while true
        status = get_param(model, 'SimulationStatus');
        if strcmpi(status, 'stopped')
            fprintf('[baseline] Simulink status: stopped.\n');
            break
        end
        if toc(lastPrint) >= 10
            try
                tNow = get_param(model, 'SimulationTime');
                fprintf('[baseline] still running: t=%.2f / %.2f days, status=%s\n', ...
                    tNow, stopDay, status);
            catch
                fprintf('[baseline] still running: status=%s\n', status);
            end
            lastPrint = tic;
        end
        pause(0.5);
    end
end
