package main.experiments.types;

import main.experiments.Experiment;
import main.experiments.ExperimentResult;
import main.models.frequency.FrequencyTable;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class PosteriorDistance extends Experiment {

    public PosteriorDistance(Instances instances1, Instances instances2, int nAttributesActive, int[] attributeIndices, double sampleScale, int nTests){
        super(instances1, instances2, nAttributesActive, attributeIndices, new int[]{instances1.classIndex()}, sampleScale, nTests);
    }

    @Override
    public ArrayList<ExperimentResult> getResults(FrequencyTable model1, FrequencyTable model2, int[] attributeSubset, double sampleScale) {
        return model1.findPosteriorDistance(model2, attributeSubset, sampleScale);
    }

    @Override
    public String[][] getResultTable() {
        return this.getResultTable(0, "*");
    }
}
