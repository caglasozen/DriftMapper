package main.models.sampling;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by Lee on 19/04/2016.
 **/
public class RandomSamples extends AbstractSampler {
    private long seed;
    private Random rng;
    private HashSet<double[]> prevInst;

    public RandomSamples(Instances dataSet, int nInstances, long seed) {
        this.seed = seed;
        this.nInstances = nInstances;
        this.setDataSet(dataSet);
        this.reset();
    }

    public RandomSamples(RandomSamples samples1, RandomSamples samples2) {
        this.seed = (samples1.getSeed() + samples2.getSeed()) / 2;
        this.nInstances = (samples1.getNInstances() + samples2.getNInstances()) / 2;
        this.setDataSet(AbstractSampler.findIntersectionBetweenInstances(samples1.getDataSet(), samples2.getDataSet()));
        this.reset();
    }

    @Override
    public void reset() {
        this.rng = new Random(this.seed);
        this.prevInst = new HashSet<>();
        this.nInstancesGeneratedSoFar = 0;
    }

    @Override
    public AbstractSampler copy(){
        return new RandomSamples(this.dataSet, this.nInstances, this.seed);
    }

    @Override
    public Instances generateAllInstances() {
        this.sampledInstances = new Instances(this.dataSet, this.nInstances);
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

    @Override
    public Instance nextInstance() {
        DenseInstance inst = new DenseInstance(this.dataSet.numAttributes());
        do {
            for (int i = 0; i < this.allPossibleValues.size(); i++) {
                ArrayList<String> attributeValues = this.allPossibleValues.get(i);
                int valueIndex = rng.nextInt(attributeValues.size());
                inst.setValue(i, Double.parseDouble(attributeValues.get(valueIndex)));
            }
        } while (this.prevInst.contains(inst.toDoubleArray()));
        this.prevInst.add(inst.toDoubleArray());
        this.nInstancesGeneratedSoFar += 1;
        return inst;
    }

    public long getSeed() {
        return this.seed;
    }
}
