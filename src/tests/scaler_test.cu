#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "training/graph_group.h"

using namespace marian;

TEST_CASE("LR Scaler behaves as expected", "[LR Scaler]")  {
    SECTION("scaling is off") {
        Scaler scaler(1700.0, 1700.0, 1, 4, 1);
        size_t tau = scaler.getNewTau();
        float batch_flex_lr = scaler.getNewBatchLR();
        REQUIRE(tau == 4);
        REQUIRE(batch_flex_lr == 1700);
        
    }
}
