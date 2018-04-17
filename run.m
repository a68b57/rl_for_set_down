close all; clearvars; clc;

% Load the Thialf standard model

thialf = hmcSpectralBody.createThialf266DeepFromNetwork;
thialf.addPOI('crane_hook',[-100, 0, 50]);

figure;
thialf.plotVessel('k-');

heave_rao = thialf.get3DRao('heave','poi','crane_hook');
waves = hmcSpectrum.createSpectrum('jonswap','Hs',2,'Tp',5,'frequency',0:0.01:4,'comingFrom',0,'gamma',3.3);

heave_response_spectrum = heave_rao.getResponse(waves,'doPlot',true);

N = 10000;
dT = 0.1; % s

[Heave, T] = heave_response_spectrum.makeTimetrace(N, dT);

plot(T,Heave);
ylabel('Vertical position of crane hook [m]');
xlabel('Time');

%% Run a simulation

time = 0;
hoist_length = 10;
i = 0;
previous_elevation = Heave(1) - hoist_length

while true
    i = i + 1;
    cargo_elevation = Heave(i) - hoist_length;
    cargo_velocity = (cargo_elevation - previous_elevation ) / dT;
    
    % determine action
    
    switch action:
        case 'lower':
            hoist_length = hoist_length + payout_speed * dT;
        otherwise:
            error('Make a decission!');
    end
    
    % do we have impact?
    
    if cargo_elevation < 0
        % we have impact!
        score = (1 - cargo_velocity)
        break;
    end
    
    if i > 1000
        score = "you are too slow!"
        break;
    end
end






