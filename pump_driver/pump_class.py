import serial
import time

"""
Software driver package for WX10Pump
"""
class WX10Pump:
    def __init__(self, port, address):
        """
        Addres of current pump. Only onw pump can be acessed
        at a time over the sereial port, but the address can
        be changed to access different pumps
        """
        self.address = address
        """
        Serial port connection for comunication with the pump.
        See the test script for guidance on how to set it up.
        """
        self.serial  = port
        
    # Methods

    def set_address(self, address):
        """
        Change the current address to acess another pump
        """
        self.address = address
        time.sleep(0.03)

    def set_speed(self, rpm, run=True, full_speed=False, cw=True):
        """
        Sets the speed of the pump with the current address
        """
        cmd = self._build_wj_command(rpm, run, full_speed, cw)
        self._send(cmd)

    def stop(self):
        """
        Stops the motor
        """
        self.set_speed(0, run=False)

    def close(self):
        """
        Closes the serial port
        """
        self.serial.close()
    
    def get_state(self):
        """
        Access the current state of the pump. Since the motors
        are open loop, the state is idential to the last command
        transmited.
        """
        pdu = [0x52, 0x4A]  #'RJ'
        length = len(pdu)
        fcs = self.address ^ length ^ pdu[0] ^ pdu[1]
        cmd = bytes([0xE9, self.address, length] + pdu + [fcs])
        response = self._send(cmd, expect_response=True, response_len=10)

        pdu_data = response[3:-1]
        speed = ((pdu_data[2] << 8) | pdu_data[3]) / 10.0
        control = pdu_data[4]
        direction = pdu_data[5]
        return {
            "speed_rpm": speed,
            "running": bool(control & 0x01),
            "full_speed": bool(control & 0x02),
            "direction": "CW" if direction & 0x01 else "CCW"
        }

    
    ## Private methods for internal use only
    def _send(self, cmd, expect_response=False, response_len=10, delay=0.03):
        """
        Transmits a command through the serial bus
        """
        self.serial.reset_input_buffer()
        self.serial.write(cmd)
        time.sleep(delay)

        if expect_response:
            time.sleep(delay*5)  # give pump time to respond
            response = self.serial.read(response_len)
            if len(response) != response_len:
                raise TimeoutError("No response or incomplete reply from pump")
            return response
        return None

    
    def _build_wj_command(self, rpm, run, full_speed, cw):
        """
        Builds the command for trasmission over the sereial bus.
        It is possible to addjust the 
        """
        rpm10 = int(rpm * 10)
        speed_H = (rpm10 >> 8) & 0xFF
        speed_L = rpm10 & 0xFF
        control = (0x01 if run else 0x00) | (0x02 if full_speed else 0x00)
        direction = 0x01 if cw else 0x00
        pdu = [0x57, 0x4A, speed_H, speed_L, control, direction]
        length = len(pdu)
        fcs = self.address ^ length
        for b in pdu:
            fcs ^= b
        return bytes([0xE9, self.address, length] + pdu + [fcs])
