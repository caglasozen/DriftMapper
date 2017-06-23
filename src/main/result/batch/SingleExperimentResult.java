package main.result.batch;

import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by loongkuan on 8/12/2016.
 **/
public class SingleExperimentResult extends ExperimentResult{
    public SingleExperimentResult(double distance, double[] separateDistance, Instances instances) {
        super(distance, separateDistance, instances);
    }

    SingleExperimentResult(ArrayList<ExperimentResult> results) {
        super(results);
    }
}
