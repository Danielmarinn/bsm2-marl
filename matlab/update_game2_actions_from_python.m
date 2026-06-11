function update_game2_actions_from_python(action_file)
% UPDATE_GAME2_ACTIONS_FROM_PYTHON
%   Read one multi-agent action row written by the Python Game2 coordinator
%   and apply all four manipulated variables to the BSM2 workspace AND to
%   the corresponding Simulink Constant blocks via set_param.
%
%   Bridge contract (post-TP-007 restoration):
%       - assignin keeps the workspace variables in sync (used by
%         RL_main_game2.m for state-csv readback).
%       - set_param writes the Constant block Value parameters, which is
%         the only path that propagates at every pause+continue in normal
%         SimulationMode (assignin alone propagates ONLY at the first pause).

    defaults = struct( ...
        'Qec',    2.0, ...
        'Qint',   61944.0, ...
        'SO4ref', 2.0, ...
        'Qw',     300.0);

    bounds = struct( ...
        'Qec',    [0.0, 5.0], ...
        'Qint',   [5000.0, 61944.0], ...
        'SO4ref', [0.0, 10.0], ...
        'Qw',     [0.0, 450.0]);

    try
        actions = readtable(action_file);
        last_row = actions(end, :);

        Qec    = game2_read_action(last_row, 'Qec',    defaults.Qec,    bounds.Qec);
        Qint   = game2_read_action(last_row, 'Qint',   defaults.Qint,   bounds.Qint);
        SO4ref = game2_read_action(last_row, 'SO4ref', defaults.SO4ref, bounds.SO4ref);
        Qw     = game2_read_action(last_row, 'Qw',     defaults.Qw,     bounds.Qw);
    catch e
        warning('update_game2_actions_from_python: %s - using BSM2 defaults', e.message);
        Qec    = defaults.Qec;
        Qint   = defaults.Qint;
        SO4ref = defaults.SO4ref;
        Qw     = defaults.Qw;
    end

    % Workspace assignin: still needed so RL_main_game2.m can read these
    % values back for state CSV / logging. Does NOT propagate to the
    % running model after the first pause.
    assignin('base', 'carb1',  Qec);
    assignin('base', 'Qintr',  Qint);
    assignin('base', 'SO4ref', SO4ref);
    assignin('base', 'Qw',     Qw);
    assignin('base', 'Qw_low', Qw);
    assignin('base', 'Qw_high', Qw);

    % Direct tunable-parameter writes to the Constant blocks identified by
    % EXP-002 and validated by EXP-004. THIS is the path that updates the
    % plant at every pause+continue. Plus a read-back guard so a silent
    % regression of this block trips an assertion next time.
    model = 'bsm2_cl';
    if bdIsLoaded(model)
        blocks = { ...
            [model '/ActivatedSludge/carb1'],                  Qec; ...
            [model '/ActivatedSludge/Qintr'],                  Qint; ...
            [model '/ActivatedSludge/SO4_ref'],                SO4ref; ...
            [model '/Settler/Qw_time_controller/Qw_winter'],   Qw; ...
            [model '/Settler/Qw_time_controller/Qw_winter1'],  Qw; ...
            [model '/Settler/Qw_time_controller/Qw_summer'],   Qw; ...
            [model '/Settler/Qw_time_controller/Qw_summer1'],  Qw};
        for k = 1:size(blocks,1)
            blk = blocks{k,1};
            val = blocks{k,2};
            try
                value_text = num2str(val, '%.15g');
                set_param(blk, 'Value', value_text);

                % Read-back guard: compare against the exact numeric value
                % encoded into the Constant block, not the pre-rounded
                % MATLAB double.
                echo_text = get_param(blk, 'Value');
                echoed = str2double(echo_text);
                target = str2double(value_text);
                tol = max(1e-6, 1e-10 * max(1, abs(target)));

                if ~isfinite(echoed) || abs(echoed - target) > tol
                    error(['BRIDGE_GUARD: set_param echo mismatch on ' blk ...
                          ' (wrote ' value_text ', got ' echo_text ').']);
                end
            catch setErr
                warning('update_game2_actions_from_python: %s', setErr.message);
            end
        end
    else
        warning('update_game2_actions_from_python: model %s not loaded; set_param skipped.', model);
    end

    fprintf('[GAME2] Qec=%.3f m3/d | Qint=%.1f m3/d | SO4ref=%.3f mg/L | Qw=%.3f m3/d\n', ...
        Qec, Qint, SO4ref, Qw);
end


function value = game2_read_action(row, name, default_value, limits)
    value = default_value;
    if any(strcmp(row.Properties.VariableNames, name))
        raw = row.(name);
        if iscell(raw)
            raw = raw{1};
        end
        raw = double(raw);
        if ~isempty(raw) && isfinite(raw(1))
            value = raw(1);
        end
    end
    value = max(limits(1), min(limits(2), value));
end
