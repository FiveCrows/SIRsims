// Create a special file to avoid conflict with R variable in cxxopts
// include file

#ifndef __STATES_H__
#define __STATES_H__

// Poor programming (GE)
// Why? Because even variables called R inside structdure
// will be changed due to preprocessing. 
// Better to use enums

//States
#define S 0
#define L 1
#define IA 2
#define PS 3
#define IS 4
#define HOME 5
#define H 6
#define ICU 7
#define R 8
// Potentially infected
#define PotL 10
// People Vaccinated get one of the following two states
#define V1 11
#define V2 12

#endif
