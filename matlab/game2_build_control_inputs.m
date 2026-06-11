function game2_build_control_inputs(csvpath)
% GAME2_BUILD_CONTROL_INPUTS  Rebuild the full-length KLa / external-carbon
% control trajectories for a Game2 closed-loop RL run and inject them into the
% base workspace as kla1in..kla5in and carbon1in..carbon5in, the variables that
% perf_plant_bsm2.m needs to compute AE / mixing / external-carbon cost.
%
% The bsm2_cl model does NOT log these actuator inputs as full trajectories
% (only ~26 samples), so perf_plant cannot read them directly. The coordinator
% training log, however, records the actually-applied KLa1..KLa5 and Qec at
% every 15-min control step. We rebuild the full vectors from that log,
% aligned (zero-order hold) onto the model output-time vector t.
%
% Run this in the SAME MATLAB session right after the co-sim finishes, then
% call the evaluator:
%   game2_build_control_inputs('...\prog_RL\logs\game2_sac_training_log.csv');
%   perf_plant_bsm2        % full 245-609 window  (or perf_plant_check to test)

    if nargin < 1 || isempty(csvpath)
        error('game2_build_control_inputs: provide the training-log CSV path.');
    end
    if evalin('base', 'exist(''t'',''var'')') ~= 1
        error('game2_build_control_inputs: base workspace has no t; run the co-sim first.');
    end

    T = readtable(csvpath);
    t = evalin('base', 't');  t = t(:);

    ctime = T.time(:);
    ok = isfinite(ctime);
    ctime = ctime(ok);

    function v = hold_onto_t(colname)
        x = T.(colname)(ok);
        x = x(:);
        good = isfinite(x);
        % zero-order hold: each action is held until the next control step
        v = interp1(ctime(good), x(good), t, 'previous', 'extrap');
        v = v(:);
    end

    kla1in = hold_onto_t('KLa1');
    kla2in = hold_onto_t('KLa2');
    kla3in = hold_onto_t('KLa3');
    kla4in = hold_onto_t('KLa4');
    kla5in = hold_onto_t('KLa5');

    carbon1in = hold_onto_t('Qec');       % external carbon flow to reactor 1 (m3/d)
    carbon2in = zeros(numel(t), 1);
    carbon3in = zeros(numel(t), 1);
    carbon4in = zeros(numel(t), 1);
    carbon5in = zeros(numel(t), 1);

    names = {'kla1in','kla2in','kla3in','kla4in','kla5in', ...
             'carbon1in','carbon2in','carbon3in','carbon4in','carbon5in'};
    vals  = {kla1in,kla2in,kla3in,kla4in,kla5in, ...
             carbon1in,carbon2in,carbon3in,carbon4in,carbon5in};
    for k = 1:numel(names)
        assignin('base', names{k}, vals{k});
    end

    fprintf('Injected %d control-input vectors (length %d) from:\n  %s\n', ...
            numel(names), numel(t), csvpath);
    fprintf('KLa4 range [%.1f .. %.1f], Qec range [%.2f .. %.2f]\n', ...
            min(kla4in), max(kla4in), min(carbon1in), max(carbon1in));
end
