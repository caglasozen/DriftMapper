package main.models.sampling;

import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by Lee on 19/04/2016.
 **/
public class RandomSamples extends AbstractSampler {
    private long seed;

    public RandomSamples(Instances dataSet, int nInstances, long seed) {
        this.seed = seed;
        this.dataSet = dataSet;
        this.nInstances = nInstances;
        this.reset();
    }

    @Override
    public AbstractSampler copy(){
        return new RandomSamples(this.dataSet, this.nInstances, this.seed);
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.dataSet = dataSet;
        this.reset();
    }

    @Override
    public Instances generateInstances() {
        this.sampledInstances = new Instances(this.dataSet, this.nInstances);
        Random rng = new Random(this.seed);
        HashSet<double[]> prevInst = new HashSet<>();

        while (sampledInstances.size() < this.nInstances) {
            DenseInstance inst = new DenseInstance(this.dataSet.numAttributes());
            for (int i = 0; i < this.allPossibleValues.size(); i++) {
                ArrayList<String> attributeValues = this.allPossibleValues.get(i);
                int valueIndex = rng.nextInt(attributeValues.size());
                inst.setValue(i, Double.parseDouble(attributeValues.get(valueIndex)));
            }
            if (!prevInst.contains(inst.toDoubleArray())) sampledInstances.add(inst);
            prevInst.add(inst.toDoubleArray());
        }
        return sampledInstances;
    }
}
