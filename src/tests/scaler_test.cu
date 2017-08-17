#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "training/graph_group.h"

using namespace marian;

TEST_CASE("LR Scaler behaves as expected", "[LR Scaler]")  {
    SECTION("scaling is off") {
        Scaler scaler(1700.0, 1700.0, 1, 4, 1);
        size_t tau = scaler.getNewTau();
        float batch_flex_lr = scaler.getNewBatchLR();
        for (int i = 0; i<10000; i++) {
            REQUIRE(tau == 4);
            REQUIRE(batch_flex_lr == 1700);
            scaler.newBatch();
            tau = scaler.getNewTau();
            batch_flex_lr = scaler.getNewBatchLR();
        }
    }

    SECTION("Scale LR") {
        Scaler scaler(1700.0, 1700.0, 1, 4, 12);
        for (int i = 0; i<4; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 1);
            REQUIRE(batch_flex_lr == 1700);
            scaler.newBatch();
        }

        for (int i = 0; i<4; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 2);
            REQUIRE(batch_flex_lr == 1700);
            scaler.newBatch();
        }

        for (int i = 0; i<4; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 3);
            REQUIRE(batch_flex_lr == 1700);
            scaler.newBatch();
        }

        for (int i = 0; i<10000; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 4);
            REQUIRE(batch_flex_lr == 1700);
            scaler.newBatch();
        }
    }

    SECTION("Scale LR big") {
        Scaler scaler(1700.0, 1700.0, 1, 4, 12000);
        for (int i = 0; i<4000; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 1);
            scaler.newBatch();
        }

        for (int i = 0; i<4000; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 2);
            scaler.newBatch();
        }

        for (int i = 0; i<4000; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 3);
            scaler.newBatch();
        }

        for (int i = 0; i<10000; i++) {
            size_t tau = scaler.getNewTau();
            float batch_flex_lr = scaler.getNewBatchLR();
            REQUIRE(tau == 4);
            scaler.newBatch();
        }
    }
}
