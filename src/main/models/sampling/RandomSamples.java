package main.models.sampling;

import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by Lee on 19/04/2016.
 **/
public class RandomSamples extends AbstractSampler {
    private int numberSamples = 0;
    private long seed;

    public RandomSamples(Instances dataSet, int numberSamples, long seed) {
        this.seed = seed;
        this.dataSet = dataSet;
        this.reset();

        int nCombinations = 0;
        this.numberSamples = numberSamples;
        for (ArrayList<String> attribute : this.allPossibleValues) {
            nCombinations *= attribute.size();
        }
        this.magnitudeScale = nCombinations / this.numberSamples;
    }

    @Override
    public AbstractSampler copy(){
        return new RandomSamples(this.dataSet, this.numberSamples, this.seed);
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.dataSet = dataSet;
        this.reset();
    }

    @Override
    public Instances generateInstances() {
        this.sampledInstances = new Instances(this.dataSet, this.numberSamples);
        Random rng = new Random(this.seed);

        while (sampledInstances.size() < this.numberSamples) {
            DenseInstance inst = new DenseInstance(this.dataSet.numAttributes());
            for (int i = 0; i < this.allPossibleValues.size(); i++) {
                ArrayList<String> attributeValues = this.allPossibleValues.get(i);
                int valueIndex = rng.nextInt(attributeValues.size());
                inst.setValue(i, Double.parseDouble(attributeValues.get(valueIndex)));
            }
            sampledInstances.add(inst);
        }
        return sampledInstances;
    }
}
