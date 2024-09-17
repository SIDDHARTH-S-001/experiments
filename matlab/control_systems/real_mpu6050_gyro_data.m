s = serial('port_number', 'BaudRate', 115200);
fopen(s);
sample_rate = 1000; % Hz
wait_time = 3*60; % secs (converting minutes to secs)
num_readings = sample_rate * wait_time;
for i = 1:num_readings
    realGyroData(i) = fscanf(s, '%d');
end

fclose(s);