package main.experiments.types;

import main.experiments.Experiment;
import main.experiments.ExperimentResult;
import main.models.frequency.FrequencyTable;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class ConditionedCovariateDistance extends Experiment {

    public ConditionedCovariateDistance(Instances instances1, Instances instances2, int nAttributesActive, int[] attributeIndices, double sampleScale, int nTests){
        super(instances1, instances2, nAttributesActive, attributeIndices, new int[]{instances1.classIndex()}, sampleScale, nTests);
    }

    @Override
    public ArrayList<ExperimentResult> getResults(FrequencyTable model1, FrequencyTable model2, int[] attributeSubset, double sampleScale) {
        return model1.findLikelihoodDistance(model2, attributeSubset, sampleScale);
    }

    @Override
    public String[][] getResultTable(){
        int nCombinations = this.resultMap.size();
        int nClasses = this.sampleInstance.numClasses();
        String[][] collatedResults = new String[nCombinations*nClasses][9];
        for (int i = 0; i < nClasses; i++) {
            String[][] tmpResult = this.getResultTable(i, this.sampleInstance.classAttribute().value(i));
            for (int j = 0; j < tmpResult.length; j++) {
                collatedResults[i*nCombinations + j] = tmpResult[j];
            }
        }
        return collatedResults;
    }

}
