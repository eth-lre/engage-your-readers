import serial 
 
ser = serial.Serial() 
ser.port = 'COM3' # replace COMx with the actual COM port name 
ser.open() 
 
# Send trial trigger code 
Trigger = 5 # trigger code must be between 1-255 
ser.write(bytes(chr(Trigger), 'UTF-8')) 
 
# End of script 
ser.close() # important before ending code!