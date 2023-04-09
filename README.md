# Hopeful Ribbon
## How the location works
Location feature, We first have a checker to determine whether or not the user inputs a valid postal code from either Canada or The US. If the postal code is valid the user then inputs their address and using the Google maps API and geocode function we determine the location of three close hospitals. If there is no hospital in a twenty kilometer radius the programs outputs "No pharmacy found within 20000 meters of the entered address." If the google maps API finds nearby hospitals it will output the name and address of said hospital.
