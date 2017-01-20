#include <iostream>
#include <vector>
#include <HFO.hpp>
#include <cstdlib>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace hfo;

// Before running this program, first Start HFO server:
// $./bin/HFO --offense-agents 1

// Server Connection Options. See printouts from bin/HFO.
feature_set_t features = HIGH_LEVEL_FEATURE_SET;
string config_dir = "../bin/teams/base/config/formations-dt";
int port = 6000;
string server_addr = "localhost";
string team_name = "base_left";
bool goalie = false;
int numAgents = 1;

// We omit PASS & CATCH & MOVE actions here
action_t HIGH_LEVEL_ACTIONS[3] = {SHOOT,DRIBBLE,PASS};

hfo::action_t getAction(const std::vector<float>& feature_vec) {
  if (feature_vec[5] == 1 ) { // Feature 5 is 1 when the player can kick the ball
   /*************Write your code here******************/
    if (numAgents == 1) {
      return HIGH_LEVEL_ACTIONS[rand() % 2];
    } else {
	// if distance from goal post is less, then goal
	// else if opponent is near by pass, else dribble
        if (feature_vec[6] < -0.60) { return HIGH_LEVEL_ACTIONS[0]; }
        else { 
   	  if (feature_vec[9] < -0.84) {return HIGH_LEVEL_ACTIONS[2]; }
          else {return HIGH_LEVEL_ACTIONS[1];}
	}     
    }
  /****************End of Code************************/
  } else {
    return MOVE;
  }
}

float getUniformNumber(const std::vector<float>& feature_vec) {
  float unums[numAgents-1];
  //Populate the list of teammate uniform numbers
  for (int i = 0; i<numAgents-1; i++) {
    unums[i] = feature_vec[9 + 3*(numAgents-1) + 3*(i+1)];
  }
  //return one at random
  return unums[rand() % (numAgents-1)];
}

void offenseAgent () {
  // Create the HFO environment
  HFOEnvironment hfo;
  // Connect to the server and request high-level feature set. See
  // manual for more information on feature sets.
  hfo.connectToServer(features, config_dir, port, server_addr,
                      team_name, goalie);
  status_t status = IN_GAME;
  for (int episode = 0; status != SERVER_DOWN; episode++) {
    status = IN_GAME;
    while (status == IN_GAME) {
      // Get the vector of state features for the current state
      const vector<float>& feature_vec = hfo.getState();
      // Perform the action
      action_t a = getAction(feature_vec);
      if (a == PASS) {
        float unum = getUniformNumber(feature_vec);
        std::cout << "PASS to " <<unum << "\n"; 
         hfo.act(a,unum);
      } else {
        hfo.act(a);
      }
      // Advance the environment and get the game status
      status = hfo.step();
    }
    // Check what the outcome of the episode was
    cout << "Episode " << episode << " ended with status: "
         << StatusToString(status) << endl;
  }
  hfo.act(QUIT);
}

int main(int argc, char** argv) {
  for(int i = 1; i < argc; i++) {
  std::string param = std::string(argv[i]);
    if(param == "--numAgents") {
      numAgents = atoi(argv[++i]);
    } else {
      std::cout << "Invalid argument";
      return 0;
    }
  }
  std::thread agentThreads[numAgents];
  for (int agent = 0; agent < numAgents; agent++) {
    agentThreads[agent] = std::thread(offenseAgent);
    usleep(500000L);
  }
  for (int agent = 0; agent < numAgents; agent++) {
    agentThreads[agent].join();
  }
  return 0;
}
