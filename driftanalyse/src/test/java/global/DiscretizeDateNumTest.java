package global;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;
import weka.core.Instance;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;

/**
 * Created by LoongKuan on 4/07/2017.
 */
public class DiscretizeDateNumTest {
    @Test
    public void testGeneral() throws Exception {
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource("../datasets/elecNormNewClean.arff");
        DiscretizeDateNum discretizeDateNum = new DiscretizeDateNum(dataSource, dataSource.getStructure());
        dataSource.reset();
        for (int i = 0; i < 100; i++) {
            Instance instance = discretizeDateNum.discretizeInstance(dataSource.nextElement(dataSource.getStructure()));
            System.out.println(Arrays.toString(instance.toDoubleArray()));
        }
    }
    @Test
    public void DiscretizeInstance() throws Exception {
    }

}