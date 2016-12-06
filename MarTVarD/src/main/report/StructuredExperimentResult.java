package main.report;

import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 6/12/2016.
 */
public class StructuredExperimentResult extends ExperimentResult{
    Map<Integer, ExperimentResult> separateExperiments;
    Map<Integer, Double> experimentsProbability;

    public StructuredExperimentResult(double weightAverageDistance, double[] separateDistance, Instances instances,
                                      Map<Integer, ExperimentResult> separateExperiments,
                                      Map<Integer, Double> experimentsProbability) {
        super(weightAverageDistance, separateDistance, instances);
        this.separateExperiments = separateExperiments;
        this.experimentsProbability = experimentsProbability;
    }

    public StructuredExperimentResult(ArrayList<ExperimentResult> resultList) {
        super(resultList);
        /*
        Set<Integer> hashes = new HashSet<>();
        Map<Integer, ArrayList<ExperimentResult>> groupedSeparateExperiments = new HashMap<>();
        for (ExperimentResult result : resultList) {
            for (Integer hash : (StructuredExperimentResult)result.)
        }
        */
    }
}
