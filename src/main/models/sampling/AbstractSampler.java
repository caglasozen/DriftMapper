package main.models.sampling;

import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Created by Lee on 19/04/2016.
 **/
public abstract class AbstractSampler {
    protected Instances dataSet;
    double magnitudeScale = 0.0f;
    ArrayList<ArrayList<String>> allPossibleValues;
    Instances sampledInstances;

    public abstract Instances generateInstances();
    public abstract AbstractSampler copy();
    public abstract void setDataSet(Instances dataSet);

    public void reset() {
        generateAllPossibleValues();
        generateInstances();

        int nCombinations = 1;
        for (ArrayList<String> attribute : this.allPossibleValues) {
            nCombinations *= attribute.size();
        }
        this.magnitudeScale = (double)nCombinations / (double)sampledInstances.size();
    }

    public double getMagnitudeScale() {
        return this.magnitudeScale;
    }

    public Instances getDataSet() {
        return dataSet;
    }

    public Instances getSampledInstances(){
        return sampledInstances;
    }

    public ArrayList<ArrayList<String>> getAllPossibleValues() {
        return allPossibleValues;
    }

    protected void generateAllPossibleValues() {
        allPossibleValues = new ArrayList<>();
        // NOTE: j represents x_n
        for (int j = 0; j < dataSet.numAttributes() - 1; j++) {
            // Initialise local variables for this loop (aka. data for this X variable)
            HashSet<String> possibleValues = new HashSet<>();
            for (int i = 0; i < dataSet.size(); i++) {
                String curKey = Double.toString(dataSet.instance(i).value(j));
                if (!possibleValues.contains(curKey)) possibleValues.add(curKey);
            }
            // Add all the keys/Possible values to allPossibleValues
            allPossibleValues.add(new ArrayList<>(possibleValues));
        }
        //Suggest Garbage collector to run
        System.gc();
    }
}
