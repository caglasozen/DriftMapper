package main.experiments.types;

import main.experiments.Experiment;
import main.experiments.ExperimentResult;
import main.models.NaiveMatrix;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class ConditionedCovariateDistance extends Experiment {

    public ConditionedCovariateDistance(Instances instances1, Instances instances2, int nAttributesActive, int[] attributeIndices, int sampleSize, int nTests){
        super(instances1, instances2, nAttributesActive, attributeIndices, new int[]{instances1.classIndex()}, sampleSize, nTests);
    }

    @Override
    public ArrayList<ExperimentResult> getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances, double sampleScale) {
        // Get the index of instances with different classes and get probabilities
        double[] p = new double[allInstances.size()];
        double[] q = new double[allInstances.size()];
        double[] separateDistance = new double[allInstances.size()];
        double[][] instanceValues = new double[allInstances.size()][allInstances.numAttributes()];
        int nClasses = allInstances.numClasses();
        ArrayList<ArrayList<Integer>> instanceIndexPerClass = new ArrayList<>();
        for (int i = 0; i < nClasses; i++) {
            instanceIndexPerClass.add(new ArrayList<>());
        }
        for (int i = 0; i < allInstances.size(); i++) {
            instanceIndexPerClass.get((int)allInstances.get(i).classValue()).add(i);
            p[i] = model1.findPvGy(allInstances.get(i));
            q[i] = model2.findPvGy(allInstances.get(i));
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            instanceValues[i] = allInstances.get(i).toDoubleArray();
        }

        ArrayList<ExperimentResult> results = new ArrayList<>();
        for (int i = 0; i < nClasses; i++) {
            int[] instancesIndex = new int[instanceIndexPerClass.get(i).size()];
            if (instancesIndex.length > 0) {
                for (int j = 0; j < instanceIndexPerClass.get(i).size(); j++) {
                    instancesIndex[j] = instanceIndexPerClass.get(i).get(j);
                }
                double[] pGy = new double[instancesIndex.length];
                double[] qGy = new double[instancesIndex.length];
                double[] separateDistanceGy = new double[instancesIndex.length];
                double[][] instanceValuesGy = new double[instancesIndex.length][allInstances.numAttributes()];
                for (int j = 0; j < instancesIndex.length; j++) {
                    int index = instancesIndex[j];
                    pGy[j] = p[index];
                    qGy[j] = q[index];
                    separateDistanceGy[j] = separateDistance[index];
                    instanceValuesGy[j] = instanceValues[index];
                }
                double finalDistance = this.distanceMetric.findDistance(pGy, qGy) * sampleScale;
                results.add(i, new ExperimentResult(finalDistance, separateDistanceGy, instanceValuesGy));
            }
            else {
                results.add(i, new ExperimentResult(Double.POSITIVE_INFINITY, null, null));
            }
        }
        return results;
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
