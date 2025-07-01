# PFASProject
Project for PFAS degredation using PINN for MPC

Current results:
- Comparison of batch traning using the new and old way of determining the concentraction of hydrated electrons
- Pump driver allowing for direct acess usgin python
- Physical setup

Current additions:
- Variable parameters have been moved to the config folder and are described using the .yaml format
- Pump driver is created as a class interface with method for wirting new commands and changeing the address of the pump to which we write.
- EKF, jacobian and kinetic model have been added ot the predictor folder, for the development of the EKF

In the making:
- EKF for sensor corection of the CINN
