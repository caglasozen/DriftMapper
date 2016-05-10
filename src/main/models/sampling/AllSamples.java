package main.models.sampling;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by Lee on 19/04/2016.
 **/
public class AllSamples extends AbstractSampler {

    public AllSamples(Instances dataSet) {
        this.setDataSet(dataSet);
        this.reset();
    }

    public AllSamples(AllSamples samples1, AllSamples samples2) {
        this.setDataSet(AbstractSampler.findIntersectionBetweenInstances(samples1.getDataSet(), samples2.getDataSet()));
        this.reset();
    }

    @Override
    public void reset() {
        this.magnitudeScale = 1.0;
        this.nInstances = this.dataSet.size();
        this.nInstancesGeneratedSoFar = 0;
    }

    @Override
    public AbstractSampler copy() {
        return new AllSamples(this.dataSet);
    }

    @Override
    public Instances generateAllInstances() {
        int nCombinations = 1;
        for (ArrayList<String> values : this.allPossibleValues) {
            nCombinations *= values.size();
        }
        this.sampledInstances = new Instances(this.dataSet, nCombinations);
        generateCombinations(0, new ArrayList<>());
        return sampledInstances;
    }

    @Override
    public Instance nextInstance() {
        DenseInstance inst = new DenseInstance(this.dataSet.numAttributes());
        int instanceIndex = this.nInstancesGeneratedSoFar;
        for (int i = 0; i < this.allPossibleValues.size(); i++) {
            ArrayList<String> attributeValues = this.allPossibleValues.get(i);
            int valueIndex = instanceIndex % attributeValues.size();
            instanceIndex = instanceIndex / attributeValues.size();
            inst.setValue(i, Double.parseDouble(attributeValues.get(valueIndex)));
        }
        this.nInstancesGeneratedSoFar += 1;
        return inst;
    }

    private void generateCombinations(int currentIndex, ArrayList<Double> auxCombination){
        if (currentIndex == this.allPossibleValues.size()){
            Instance inst = new DenseInstance(dataSet.numAttributes());
            inst.setDataset(dataSet);
            for (int j = 0; j < auxCombination.size(); j++) {
                inst.setValue(j, auxCombination.get(j));
            }
            this.sampledInstances.add(inst);
        }
        else {
            ArrayList<String> curValues = this.allPossibleValues.get(currentIndex);
            for (String value : curValues){
                // Add the new possible x value to the list of values and call itself with the new list
                auxCombination.add(Double.parseDouble(value));
                // Add the generated combinations with the new list to the existing combinations
                generateCombinations(currentIndex + 1, auxCombination);
                // Remove the added x value from the list of values
                auxCombination.remove(auxCombination.size() - 1);
            }
        }
    }
}
