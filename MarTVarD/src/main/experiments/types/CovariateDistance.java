package main.experiments.types;

import main.experiments.Experiment;
import main.experiments.ExperimentResult;
import main.models.frequency.FrequencyTable;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class CovariateDistance extends Experiment {

    public CovariateDistance(Instances instances1, Instances instances2, int nAttributesActive, int[] attributeIndices, double sampleScale, int nTests){
        // List of 0 to n where n is the number of attributes
        super(instances1, instances2, nAttributesActive, attributeIndices, new int[]{}, sampleScale, nTests);
    }

    @Override
    public ArrayList<ExperimentResult> getResults(FrequencyTable model1, FrequencyTable model2, int[] attributeSubset, double sampleScale) {
        return model1.findCovariateDistance(model2, attributeSubset, sampleScale);
    }

    @Override
    public String[][] getResultTable() {
        return this.getResultTable(0, "NA");
    }
}
