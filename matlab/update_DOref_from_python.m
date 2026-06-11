function update_DOref_from_python(action_file)
% UPDATE_DOREF_FROM_PYTHON
%   Read the dissolved oxygen setpoint (DOref) from the action.csv written
%   by Python and apply it to the SO4ref variable in the MATLAB workspace.
%
%   Naming note: the CSV column is named 'Qec' for historical
%   compatibility of the shared communication file. The value is
%   interpreted here as SO4ref / DOref of the official oxygen loop of
%   aerobic reactor 4, NOT as external carbon.
%
%   SO4ref is the canonical BSM2 name for the dissolved oxygen setpoint
%   consumed by the SO4_control block in the ActivatedSludge subsystem.
%   In the official benchmark, that loop regulates reactor 4; the same KLa
%   signal affects reactor 5 via KLa4/2.
%
%   Limits used for CTRL-3:
%       DOref in [0, 10] mg/L
%
%   BSM2 default (Section 9, "Initialization"): DOref = 2.0 mg/L.

    DOREF_MIN = 0.0;
    DOREF_MAX = 10.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        DOref = data(end);   % 'Qec' column by historical compatibility
        DOref = max(DOREF_MIN, min(DOREF_MAX, DOref));
    catch e
        warning('update_DOref_from_python: %s - using BSM2 default (2.0 mg/L)', e.message);
        DOref = 2.0;
    end

    % Apply to the official SO4_control loop (reactor 4; reactor 5 coupled via KLa4/2)
    assignin('base', 'SO4ref', DOref);

    fprintf('[RL] DOref = %.3f mg/L (SO4ref)\n', DOref);

end
