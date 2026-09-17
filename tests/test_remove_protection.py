#!/usr/bin/env python3
"""
Test suite to validate the protection against double removal in playground.remove()
Tests the elegant solution: if not entity.removed check before _remove_from_space()
"""
import pathlib
import sys
from typing import List, Type

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData


class DummyDrone(DroneAbstract):
    """Simple drone for testing"""
    def define_message_for_all(self):
        pass
    
    def control(self):
        pass


class MyMapTest(MapAbstract):
    """Test map with basic playground setup"""
    def __init__(self, drone_type: Type[DroneAbstract] = DummyDrone, with_disposal_center: bool = True):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (1000, 1000)
        self._number_drones = 0
        self._drones: List[DroneAbstract] = []

        self._playground = ClosedPlayground(size=self._size_area)

        # Add disposal center if needed
        if with_disposal_center:
            self._disposal_center = DisposalCenter(size=(100, 100))
            self._playground.add(self._disposal_center, ((0, 0), 0))
        else:
            self._disposal_center = None


def test_1_single_remove():
    """Test 1: Simple removal without definitive"""
    print("\n" + "="*80)
    print("TEST 1: Simple removal (definitive=False)")
    print("="*80)
    
    the_map = MyMapTest()
    playground = the_map.playground

    wp = Bomb(disposal_center=the_map.disposal_center)
    playground.add(wp, ((100, 100), 0))
    
    print(f"✓ Created playground with 1 bomb")
    print(f"  - Elements in _elements: {len(playground._elements)}")
    print(f"  - Active elements: {len([e for e in playground._elements if not e.removed])}")
    
    # Remove without definitive
    playground.remove(wp, definitive=False)
    
    print(f"✓ Removed bomb (definitive=False)")
    print(f"  - Elements in _elements: {len(playground._elements)}")
    print(f"  - Active elements: {len([e for e in playground._elements if not e.removed])}")
    print(f"  - Removed elements: {len([e for e in playground._elements if e.removed])}")
    print(f"  - wp.removed = {wp.removed}")
    
    assert wp.removed == True, "Entity should be marked as removed"
    assert wp in playground._elements, "Entity should still be in _elements"
    
    print("✓ TEST 1 PASSED")


def test_2_double_remove_same_entity():
    """Test 2: Try to remove the same entity twice (the critical test!)"""
    print("\n" + "="*80)
    print("TEST 2: Double removal protection (THE CRITICAL TEST)")
    print("="*80)
    
    # Setup
    the_map = MyMapTest()
    playground = the_map.playground

    wp = Bomb(disposal_center=the_map.disposal_center)
    playground.add(wp, ((100, 100), 0))

    # First removal
    playground.remove(wp, definitive=False)
    print(f"✓ First removal done: wp.removed = {wp.removed}")

    print(f"\n→ Attempting to remove the same entity again...")
    print(f"  - wp.removed before 2nd remove: {wp.removed}")
    
    try:
        # This should NOT crash thanks to: if not entity.removed
        playground.remove(wp, definitive=False)
        print(f"✓ Second remove() succeeded without crash!")
        print(f"  - wp.removed after 2nd remove: {wp.removed}")
        print("✓ TEST 2 PASSED - The 'if not entity.removed' protection works!")
    except AssertionError as e:
        if "shape not in space" in str(e):
            print(f"✗ TEST 2 FAILED - Got 'shape not in space' error!")
            print(f"  Error: {e}")
            raise
        else:
            raise
    
    assert wp.removed == True, "Entity should still be marked as removed"


def test_3_cleanup_after_rescue():
    """Test 3: Simulate rescue scenario and cleanup (real-world case)"""
    print("\n" + "="*80)
    print("TEST 3: Rescue scenario + cleanup (REAL-WORLD CASE)")
    print("="*80)
    
    the_map = MyMapTest()
    playground = the_map.playground

    # Create 3 bombs
    bombs = []
    for i in range(3):
        wp = Bomb(disposal_center=the_map.disposal_center)
        bombs.append(wp)
        playground.add(wp, ((100 * (i+1), 100), 0))
    
    print(f"✓ Created playground with 3 bombs")
    print(f"  - Total elements in _elements: {len(playground._elements)}")
    
    # Simulate rescuing 2 bombs (removed without definitive)
    print(f"\n→ Simulating rescue of 2 bombs...")
    playground.remove(bombs[0], definitive=False)
    playground.remove(bombs[1], definitive=False)
    
    print(f"✓ Rescued 2 bombs")
    print(f"  - Elements in _elements: {len(playground._elements)}")
    print(f"  - Active elements: {len([e for e in playground._elements if not e.removed])}")
    print(f"  - Removed elements: {len([e for e in playground._elements if e.removed])}")
    
    # Now cleanup (this calls remove with definitive=True on all entities)
    print(f"\n→ Calling cleanup() which will call remove(definitive=True) on all...")
    try:
        playground.cleanup()
        print(f"✓ Cleanup succeeded!")
        print(f"  - Elements remaining: {len(playground._elements)}")
        print(f"  - _uids_to_entities: {len(playground._uids_to_entities)}")
        print(f"  - _shapes_to_entities: {len(playground._shapes_to_entities)}")
        print("✓ TEST 3 PASSED - Cleanup handles already-removed entities!")
    except Exception as e:
        print(f"✗ TEST 3 FAILED with error: {e}")
        raise


def test_4_drone_collision_scenario():
    """Test 4: Simulate entity removed with definitive=True then cleanup"""
    print("\n" + "="*80)
    print("TEST 4: Definitive removal + cleanup scenario")
    print("="*80)
    
    the_map = MyMapTest()
    playground = the_map.playground

    wp = Bomb(disposal_center=the_map.disposal_center)
    playground.add(wp, ((100, 100), 0))

    print(f"✓ Created playground with 1 bomb")
    print(f"  - Elements in playground: {len(playground._elements)}")
    
    # Simulate entity being destroyed definitively
    print(f"\n→ Removing with definitive=True (simulating destruction)...")
    playground.remove(wp, definitive=True)

    print(f"✓ Entity removed (definitive=True)")
    print(f"  - wp.removed: {wp.removed}")
    print(f"  - Elements in _elements: {len(playground._elements)}")
    print(f"  - wp still in _elements: {wp in playground._elements}")

    # Now cleanup
    print(f"\n→ Calling cleanup()...")
    try:
        playground.cleanup()
        print(f"✓ Cleanup succeeded!")
        print(f"  - Elements remaining: {len(playground._elements)}")
        print("✓ TEST 4 PASSED - Cleanup handles definitively-removed entities!")
    except Exception as e:
        print(f"✗ TEST 4 FAILED with error: {e}")
        raise


def test_5_mixed_scenario():
    """Test 5: Mix of removed and active entities during cleanup"""
    print("\n" + "="*80)
    print("TEST 5: Mixed scenario - some removed, some active")
    print("="*80)
    
    the_map = MyMapTest()
    playground = the_map.playground

    # Create 5 bombs
    bombs = []
    for i in range(5):
        wp = Bomb(disposal_center=the_map.disposal_center)
        bombs.append(wp)
        playground.add(wp, ((100 * (i+1), 100), 0))
    
    print(f"✓ Created playground with 5 bombs")
    
    # Remove 2 without definitive (simulating rescue)
    playground.remove(bombs[0], definitive=False)
    playground.remove(bombs[2], definitive=False)
    
    print(f"✓ Rescued 2 bombs (removed=True, but still in _elements)")
    print(f"  - Active: {len([e for e in playground._elements if not e.removed])}")
    print(f"  - Removed: {len([e for e in playground._elements if e.removed])}")
    
    # Remove 1 definitively (simulating destruction)
    playground.remove(bombs[4], definitive=True)
    
    print(f"✓ Destroyed 1 bomb (removed definitively)")
    print(f"  - Total in _elements: {len(playground._elements)}")
    print(f"  - Active: {len([e for e in playground._elements if not e.removed])}")
    
    # Now cleanup
    print(f"\n→ Calling cleanup() on mixed state...")
    try:
        playground.cleanup()
        print(f"✓ Cleanup succeeded!")
        print(f"  - Elements remaining: {len(playground._elements)}")
        print("✓ TEST 5 PASSED - Cleanup handles mixed scenarios!")
    except Exception as e:
        print(f"✗ TEST 5 FAILED with error: {e}")
        raise


def test_6_triple_remove_attempt():
    """Test 6: Try to remove the same entity THREE times"""
    print("\n" + "="*80)
    print("TEST 6: Triple removal attempt (stress test)")
    print("="*80)
    
    the_map = MyMapTest()
    playground = the_map.playground

    wp = Bomb(disposal_center=the_map.disposal_center)
    playground.add(wp, ((100, 100), 0))
    
    print(f"✓ Created bomb")
    
    # First remove
    playground.remove(wp, definitive=False)
    print(f"✓ First remove: wp.removed = {wp.removed}")
    
    # Second remove (should be protected)
    try:
        playground.remove(wp, definitive=False)
        print(f"✓ Second remove: wp.removed = {wp.removed}")
    except Exception as e:
        print(f"✗ Second remove failed: {e}")
        raise
    
    # Third remove (should still be protected)
    try:
        playground.remove(wp, definitive=False)
        print(f"✓ Third remove: wp.removed = {wp.removed}")
        print("✓ TEST 6 PASSED - Protection works for multiple attempts!")
    except Exception as e:
        print(f"✗ Third remove failed: {e}")
        raise


def test_13_removal_and_cleanup():
    """Test 13: Removal followed by cleanup (ensure no crashes)"""
    print("\n" + "="*80)
    print("TEST 13: Removal and immediate cleanup")
    print("="*80)

    the_map = MyMapTest()
    playground = the_map.playground

    # Create 3 bombs
    bombs = []
    for i in range(3):
        wp = Bomb(disposal_center=the_map.disposal_center)
        bombs.append(wp)
        playground.add(wp, ((100 * (i+1), 100), 0))

    print(f"✓ Created playground with 3 bombs")

    # Remove 2 persons (without definitive)
    playground.remove(bombs[0], definitive=False)
    playground.remove(bombs[1], definitive=False)

    print(f"✓ Removed 2 bombs (definitive=False)")

    # Now cleanup
    print(f"\n→ Calling cleanup()...")
    try:
        playground.cleanup()
        print(f"✓ Cleanup succeeded!")
        print(f"  - Elements remaining: {len(playground._elements)}")
        print("✓ TEST 13 PASSED - Removal followed by cleanup works!")
    except Exception as e:
        print(f"✗ TEST 13 FAILED with error: {e}")
        raise


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("Testing: if not entity.removed protection in playground.remove()")
    print("="*80)
    
    # Note: Tests must be run in order where cleanup tests come last
    # because cleanup() closes the arcade window
    tests = [
        ("Double removal protection", test_2_double_remove_same_entity),
        ("Triple removal stress test", test_6_triple_remove_attempt),
        ("Mixed scenario", test_5_mixed_scenario),
        ("Rescue + cleanup scenario", test_3_cleanup_after_rescue),
        ("Removal and immediate cleanup", test_13_removal_and_cleanup),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗✗✗ {test_name} FAILED ✗✗✗")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The solution 'if not entity.removed' works perfectly!")
        print("\nKey validations:")
        print("  ✓ Double removal protection works")
        print("  ✓ Triple removal protection works")
        print("  ✓ Cleanup handles mixed scenarios (removed + active)")
        print("  ✓ Rescue scenario + cleanup works correctly")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
